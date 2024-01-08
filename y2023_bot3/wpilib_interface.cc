#include <unistd.h>

#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "ctre/phoenix/CANifier.h"

#include "frc971/wpilib/ahal/AnalogInput.h"
#include "frc971/wpilib/ahal/Counter.h"
#include "frc971/wpilib/ahal/DigitalGlitchFilter.h"
#include "frc971/wpilib/ahal/DriverStation.h"
#include "frc971/wpilib/ahal/Encoder.h"
#include "frc971/wpilib/ahal/Servo.h"
#include "frc971/wpilib/ahal/TalonFX.h"
#include "frc971/wpilib/ahal/VictorSP.h"
#undef ERROR

#include "ctre/phoenix/cci/Diagnostics_CCI.h"
#include "ctre/phoenix6/TalonFX.hpp"

#include "aos/commonmath.h"
#include "aos/containers/sized_array.h"
#include "aos/events/event_loop.h"
#include "aos/events/shm_event_loop.h"
#include "aos/init.h"
#include "aos/logging/logging.h"
#include "aos/realtime.h"
#include "aos/time/time.h"
#include "aos/util/log_interval.h"
#include "aos/util/phased_loop.h"
#include "aos/util/wrapping_counter.h"
#include "frc971/autonomous/auto_mode_generated.h"
#include "frc971/can_configuration_generated.h"
#include "frc971/control_loops/drivetrain/drivetrain_can_position_generated.h"
#include "frc971/control_loops/drivetrain/drivetrain_position_generated.h"
#include "frc971/input/robot_state_generated.h"
#include "frc971/queues/gyro_generated.h"
#include "frc971/wpilib/ADIS16448.h"
#include "frc971/wpilib/buffered_pcm.h"
#include "frc971/wpilib/buffered_solenoid.h"
#include "frc971/wpilib/dma.h"
#include "frc971/wpilib/drivetrain_writer.h"
#include "frc971/wpilib/encoder_and_potentiometer.h"
#include "frc971/wpilib/joystick_sender.h"
#include "frc971/wpilib/logging_generated.h"
#include "frc971/wpilib/loop_output_handler.h"
#include "frc971/wpilib/pdp_fetcher.h"
#include "frc971/wpilib/sensor_reader.h"
#include "frc971/wpilib/wpilib_robot_base.h"
#include "y2023_bot3/constants.h"
#include "y2023_bot3/control_loops/superstructure/led_indicator.h"
#include "y2023_bot3/control_loops/superstructure/superstructure_output_generated.h"
#include "y2023_bot3/control_loops/superstructure/superstructure_position_generated.h"

DEFINE_bool(ctre_diag_server, false,
            "If true, enable the diagnostics server for interacting with "
            "devices on the CAN bus using Phoenix Tuner");

using ::aos::monotonic_clock;
using ::y2023_bot3::constants::Values;
namespace superstructure = ::y2023_bot3::control_loops::superstructure;
namespace drivetrain = ::y2023_bot3::control_loops::drivetrain;
namespace chrono = ::std::chrono;
using std::make_unique;

namespace y2023_bot3 {
namespace wpilib {
namespace {

constexpr double kMaxBringupPower = 12.0;

// TODO(Brian): Fix the interpretation of the result of GetRaw here and in the
// DMA stuff and then removing the * 2.0 in *_translate.
// The low bit is direction.

double drivetrain_velocity_translate(double in) {
  return (((1.0 / in) / Values::kDrivetrainCyclesPerRevolution()) *
          (2.0 * M_PI)) *
         Values::kDrivetrainEncoderRatio() *
         control_loops::drivetrain::kWheelRadius;
}

constexpr double kMaxFastEncoderPulsesPerSecond = std::max({
    Values::kMaxDrivetrainEncoderPulsesPerSecond(),
});
static_assert(kMaxFastEncoderPulsesPerSecond <= 1300000,
              "fast encoders are too fast");

}  // namespace

static constexpr int kCANFalconCount = 6;
static constexpr units::frequency::hertz_t kCANUpdateFreqHz = 200_Hz;

class Falcon {
 public:
  Falcon(int device_id, std::string canbus,
         std::vector<ctre::phoenix6::BaseStatusSignal *> *signals)
      : talon_(device_id, canbus),
        device_id_(device_id),
        device_temp_(talon_.GetDeviceTemp()),
        supply_voltage_(talon_.GetSupplyVoltage()),
        supply_current_(talon_.GetSupplyCurrent()),
        torque_current_(talon_.GetTorqueCurrent()),
        position_(talon_.GetPosition()),
        duty_cycle_(talon_.GetDutyCycle()) {
    // device temp is not timesynced so don't add it to the list of signals
    device_temp_.SetUpdateFrequency(kCANUpdateFreqHz);

    CHECK_NOTNULL(signals);

    supply_voltage_.SetUpdateFrequency(kCANUpdateFreqHz);
    signals->push_back(&supply_voltage_);

    supply_current_.SetUpdateFrequency(kCANUpdateFreqHz);
    signals->push_back(&supply_current_);

    torque_current_.SetUpdateFrequency(kCANUpdateFreqHz);
    signals->push_back(&torque_current_);

    position_.SetUpdateFrequency(kCANUpdateFreqHz);
    signals->push_back(&position_);

    duty_cycle_.SetUpdateFrequency(kCANUpdateFreqHz);
    signals->push_back(&duty_cycle_);
  }

  void PrintConfigs() {
    ctre::phoenix6::configs::TalonFXConfiguration configuration;
    ctre::phoenix::StatusCode status =
        talon_.GetConfigurator().Refresh(configuration);
    if (!status.IsOK()) {
      AOS_LOG(ERROR, "Failed to get falcon configuration: %s: %s",
              status.GetName(), status.GetDescription());
    }
    AOS_LOG(INFO, "configuration: %s", configuration.ToString().c_str());
  }

  void WriteConfigs(ctre::phoenix6::signals::InvertedValue invert) {
    inverted_ = invert;

    ctre::phoenix6::configs::CurrentLimitsConfigs current_limits;
    current_limits.StatorCurrentLimit =
        constants::Values::kDrivetrainStatorCurrentLimit();
    current_limits.StatorCurrentLimitEnable = true;
    current_limits.SupplyCurrentLimit =
        constants::Values::kDrivetrainSupplyCurrentLimit();
    current_limits.SupplyCurrentLimitEnable = true;

    ctre::phoenix6::configs::MotorOutputConfigs output_configs;
    output_configs.NeutralMode =
        ctre::phoenix6::signals::NeutralModeValue::Brake;
    output_configs.DutyCycleNeutralDeadband = 0;

    output_configs.Inverted = inverted_;

    ctre::phoenix6::configs::TalonFXConfiguration configuration;
    configuration.CurrentLimits = current_limits;
    configuration.MotorOutput = output_configs;

    ctre::phoenix::StatusCode status =
        talon_.GetConfigurator().Apply(configuration);
    if (!status.IsOK()) {
      AOS_LOG(ERROR, "Failed to set falcon configuration: %s: %s",
              status.GetName(), status.GetDescription());
    }

    PrintConfigs();
  }

  ctre::phoenix6::hardware::TalonFX *talon() { return &talon_; }

  flatbuffers::Offset<frc971::control_loops::CANFalcon> WritePosition(
      flatbuffers::FlatBufferBuilder *fbb) {
    frc971::control_loops::CANFalcon::Builder builder(*fbb);
    builder.add_id(device_id_);
    builder.add_device_temp(device_temp());
    builder.add_supply_voltage(supply_voltage());
    builder.add_supply_current(supply_current());
    builder.add_torque_current(torque_current());
    builder.add_duty_cycle(duty_cycle());

    double invert =
        (inverted_ == ctre::phoenix6::signals::InvertedValue::Clockwise_Positive
             ? 1
             : -1);

    builder.add_position(
        constants::Values::DrivetrainCANEncoderToMeters(position()) * invert);

    return builder.Finish();
  }

  int device_id() const { return device_id_; }
  float device_temp() const { return device_temp_.GetValue().value(); }
  float supply_voltage() const { return supply_voltage_.GetValue().value(); }
  float supply_current() const { return supply_current_.GetValue().value(); }
  float torque_current() const { return torque_current_.GetValue().value(); }
  float duty_cycle() const { return duty_cycle_.GetValue().value(); }
  float position() const { return position_.GetValue().value(); }

  // returns the monotonic timestamp of the latest timesynced reading in the
  // timebase of the the syncronized CAN bus clock.
  int64_t GetTimestamp() {
    std::chrono::nanoseconds latest_timestamp =
        torque_current_.GetTimestamp().GetTime();

    return latest_timestamp.count();
  }

  void RefreshNontimesyncedSignals() { device_temp_.Refresh(); };

 private:
  ctre::phoenix6::hardware::TalonFX talon_;
  int device_id_;

  ctre::phoenix6::signals::InvertedValue inverted_;

  ctre::phoenix6::StatusSignal<units::temperature::celsius_t> device_temp_;
  ctre::phoenix6::StatusSignal<units::voltage::volt_t> supply_voltage_;
  ctre::phoenix6::StatusSignal<units::current::ampere_t> supply_current_,
      torque_current_;
  ctre::phoenix6::StatusSignal<units::angle::turn_t> position_;
  ctre::phoenix6::StatusSignal<units::dimensionless::scalar_t> duty_cycle_;
};

class CANSensorReader {
 public:
  CANSensorReader(
      aos::EventLoop *event_loop,
      std::vector<ctre::phoenix6::BaseStatusSignal *> signals_registry)
      : event_loop_(event_loop),
        signals_(signals_registry.begin(), signals_registry.end()),
        can_position_sender_(
            event_loop
                ->MakeSender<frc971::control_loops::drivetrain::CANPosition>(
                    "/drivetrain")) {
    event_loop->SetRuntimeRealtimePriority(40);
    event_loop->SetRuntimeAffinity(aos::MakeCpusetFromCpus({1}));
    timer_handler_ = event_loop->AddTimer([this]() { Loop(); });
    timer_handler_->set_name("CANSensorReader Loop");

    event_loop->OnRun([this]() {
      timer_handler_->Schedule(event_loop_->monotonic_now(),
                               1 / kCANUpdateFreqHz);
    });
  }

  void set_falcons(std::shared_ptr<Falcon> right_front,
                   std::shared_ptr<Falcon> right_back,
                   std::shared_ptr<Falcon> left_front,
                   std::shared_ptr<Falcon> left_back) {
    right_front_ = std::move(right_front);
    right_back_ = std::move(right_back);
    left_front_ = std::move(left_front);
    left_back_ = std::move(left_back);
  }

 private:
  void Loop() {
    ctre::phoenix::StatusCode status =
        ctre::phoenix6::BaseStatusSignal::WaitForAll(2000_ms, signals_);

    if (!status.IsOK()) {
      AOS_LOG(ERROR, "Failed to read signals from falcons: %s: %s",
              status.GetName(), status.GetDescription());
    }

    auto builder = can_position_sender_.MakeBuilder();

    for (auto falcon : {right_front_, right_back_, left_front_, left_back_}) {
      falcon->RefreshNontimesyncedSignals();
    }

    aos::SizedArray<flatbuffers::Offset<frc971::control_loops::CANFalcon>,
                    kCANFalconCount>
        falcons;

    for (auto falcon : {right_front_, right_back_, left_front_, left_back_}) {
      falcons.push_back(falcon->WritePosition(builder.fbb()));
    }

    auto falcons_list =
        builder.fbb()
            ->CreateVector<
                flatbuffers::Offset<frc971::control_loops::CANFalcon>>(falcons);

    frc971::control_loops::drivetrain::CANPosition::Builder
        can_position_builder =
            builder
                .MakeBuilder<frc971::control_loops::drivetrain::CANPosition>();

    can_position_builder.add_falcons(falcons_list);
    can_position_builder.add_timestamp(right_front_->GetTimestamp());
    can_position_builder.add_status(static_cast<int>(status));

    builder.CheckOk(builder.Send(can_position_builder.Finish()));
  }

  aos::EventLoop *event_loop_;

  const std::vector<ctre::phoenix6::BaseStatusSignal *> signals_;
  aos::Sender<frc971::control_loops::drivetrain::CANPosition>
      can_position_sender_;

  std::shared_ptr<Falcon> right_front_, right_back_, left_front_, left_back_;

  // Pointer to the timer handler used to modify the wakeup.
  ::aos::TimerHandler *timer_handler_;
};

// Class to send position messages with sensor readings to our loops.
class SensorReader : public ::frc971::wpilib::SensorReader {
 public:
  SensorReader(::aos::ShmEventLoop *event_loop,
               std::shared_ptr<const Values> values,
               CANSensorReader *can_sensor_reader)
      : ::frc971::wpilib::SensorReader(event_loop),
        values_(std::move(values)),
        auto_mode_sender_(
            event_loop->MakeSender<::frc971::autonomous::AutonomousMode>(
                "/autonomous")),
        superstructure_position_sender_(
            event_loop->MakeSender<superstructure::Position>(
                "/superstructure")),
        drivetrain_position_sender_(
            event_loop
                ->MakeSender<::frc971::control_loops::drivetrain::Position>(
                    "/drivetrain")),
        gyro_sender_(event_loop->MakeSender<::frc971::sensors::GyroReading>(
            "/drivetrain")),
        can_sensor_reader_(can_sensor_reader) {
    // Set to filter out anything shorter than 1/4 of the minimum pulse width
    // we should ever see.
    UpdateFastEncoderFilterHz(kMaxFastEncoderPulsesPerSecond);
    event_loop->SetRuntimeAffinity(aos::MakeCpusetFromCpus({0}));
  }

  void Start() override { AddToDMA(&imu_yaw_rate_reader_); }

  // Auto mode switches.
  void set_autonomous_mode(int i, ::std::unique_ptr<frc::DigitalInput> sensor) {
    autonomous_modes_.at(i) = ::std::move(sensor);
  }

  void set_yaw_rate_input(::std::unique_ptr<frc::DigitalInput> sensor) {
    imu_yaw_rate_input_ = ::std::move(sensor);
    imu_yaw_rate_reader_.set_input(imu_yaw_rate_input_.get());
  }

  void RunIteration() override {
    superstructure_reading_->Set(true);
    {
      auto builder = superstructure_position_sender_.MakeBuilder();

      superstructure::Position::Builder position_builder =
          builder.MakeBuilder<superstructure::Position>();
      builder.CheckOk(builder.Send(position_builder.Finish()));
    }

    {
      auto builder = drivetrain_position_sender_.MakeBuilder();
      frc971::control_loops::drivetrain::Position::Builder drivetrain_builder =
          builder.MakeBuilder<frc971::control_loops::drivetrain::Position>();
      drivetrain_builder.add_left_encoder(
          -constants::Values::DrivetrainEncoderToMeters(
              drivetrain_left_encoder_->GetRaw()));
      drivetrain_builder.add_left_speed(
          drivetrain_velocity_translate(drivetrain_left_encoder_->GetPeriod()));

      drivetrain_builder.add_right_encoder(
          constants::Values::DrivetrainEncoderToMeters(
              drivetrain_right_encoder_->GetRaw()));
      drivetrain_builder.add_right_speed(-drivetrain_velocity_translate(
          drivetrain_right_encoder_->GetPeriod()));

      builder.CheckOk(builder.Send(drivetrain_builder.Finish()));
    }

    {
      auto builder = gyro_sender_.MakeBuilder();
      ::frc971::sensors::GyroReading::Builder gyro_reading_builder =
          builder.MakeBuilder<::frc971::sensors::GyroReading>();
      // +/- 2000 deg / sec
      constexpr double kMaxVelocity = 4000;  // degrees / second
      constexpr double kVelocityRadiansPerSecond =
          kMaxVelocity / 360 * (2.0 * M_PI);

      // Only part of the full range is used to prevent being 100% on or off.
      constexpr double kScaledRangeLow = 0.1;
      constexpr double kScaledRangeHigh = 0.9;

      constexpr double kPWMFrequencyHz = 200;
      double velocity_duty_cycle =
          imu_yaw_rate_reader_.last_width() * kPWMFrequencyHz;

      constexpr double kDutyCycleScale =
          1 / (kScaledRangeHigh - kScaledRangeLow);
      // scale from 0.1 - 0.9 to 0 - 1
      double rescaled_velocity_duty_cycle =
          (velocity_duty_cycle - kScaledRangeLow) * kDutyCycleScale;

      if (!std::isnan(rescaled_velocity_duty_cycle)) {
        gyro_reading_builder.add_velocity((rescaled_velocity_duty_cycle - 0.5) *
                                          kVelocityRadiansPerSecond);
      }
      builder.CheckOk(builder.Send(gyro_reading_builder.Finish()));
    }

    {
      auto builder = auto_mode_sender_.MakeBuilder();

      uint32_t mode = 0;
      for (size_t i = 0; i < autonomous_modes_.size(); ++i) {
        if (autonomous_modes_[i] && autonomous_modes_[i]->Get()) {
          mode |= 1 << i;
        }
      }

      auto auto_mode_builder =
          builder.MakeBuilder<frc971::autonomous::AutonomousMode>();

      auto_mode_builder.add_mode(mode);

      builder.CheckOk(builder.Send(auto_mode_builder.Finish()));
    }
  }

  std::shared_ptr<frc::DigitalOutput> superstructure_reading_;

  void set_superstructure_reading(
      std::shared_ptr<frc::DigitalOutput> superstructure_reading) {
    superstructure_reading_ = superstructure_reading;
  }

 private:
  std::shared_ptr<const Values> values_;

  aos::Sender<frc971::autonomous::AutonomousMode> auto_mode_sender_;
  aos::Sender<superstructure::Position> superstructure_position_sender_;
  aos::Sender<frc971::control_loops::drivetrain::Position>
      drivetrain_position_sender_;
  ::aos::Sender<::frc971::sensors::GyroReading> gyro_sender_;

  std::array<std::unique_ptr<frc::DigitalInput>, 2> autonomous_modes_;

  std::unique_ptr<frc::DigitalInput> imu_yaw_rate_input_;

  frc971::wpilib::DMAPulseWidthReader imu_yaw_rate_reader_;

  CANSensorReader *can_sensor_reader_;
};
class DrivetrainWriter : public ::frc971::wpilib::LoopOutputHandler<
                             ::frc971::control_loops::drivetrain::Output> {
 public:
  DrivetrainWriter(::aos::EventLoop *event_loop)
      : ::frc971::wpilib::LoopOutputHandler<
            ::frc971::control_loops::drivetrain::Output>(event_loop,
                                                         "/drivetrain") {
    event_loop->SetRuntimeRealtimePriority(
        constants::Values::kDrivetrainWriterPriority);

    event_loop->OnRun([this]() { WriteConfigs(); });
  }

  void set_falcons(std::shared_ptr<Falcon> right_front,
                   std::shared_ptr<Falcon> right_back,
                   std::shared_ptr<Falcon> left_front,
                   std::shared_ptr<Falcon> left_back) {
    right_front_ = std::move(right_front);
    right_back_ = std::move(right_back);
    left_front_ = std::move(left_front);
    left_back_ = std::move(left_back);
  }

  void set_right_inverted(ctre::phoenix6::signals::InvertedValue invert) {
    right_inverted_ = invert;
  }

  void set_left_inverted(ctre::phoenix6::signals::InvertedValue invert) {
    left_inverted_ = invert;
  }

  void HandleCANConfiguration(const frc971::CANConfiguration &configuration) {
    for (auto falcon : {right_front_, right_back_, left_front_, left_back_}) {
      falcon->PrintConfigs();
    }
    if (configuration.reapply()) {
      WriteConfigs();
    }
  }

 private:
  void WriteConfigs() {
    for (auto falcon : {right_front_.get(), right_back_.get()}) {
      falcon->WriteConfigs(right_inverted_);
    }

    for (auto falcon : {left_front_.get(), left_back_.get()}) {
      falcon->WriteConfigs(left_inverted_);
    }
  }

  void Write(
      const ::frc971::control_loops::drivetrain::Output &output) override {
    ctre::phoenix6::controls::DutyCycleOut left_control(
        SafeSpeed(output.left_voltage()));
    left_control.UpdateFreqHz = 0_Hz;
    left_control.EnableFOC = true;

    ctre::phoenix6::controls::DutyCycleOut right_control(
        SafeSpeed(output.right_voltage()));
    right_control.UpdateFreqHz = 0_Hz;
    right_control.EnableFOC = true;

    for (auto falcon : {left_front_.get(), left_back_.get()}) {
      ctre::phoenix::StatusCode status =
          falcon->talon()->SetControl(left_control);

      if (!status.IsOK()) {
        AOS_LOG(ERROR, "Failed to write control to falcon: %s: %s",
                status.GetName(), status.GetDescription());
      }
    }

    for (auto falcon : {right_front_.get(), right_back_.get()}) {
      ctre::phoenix::StatusCode status =
          falcon->talon()->SetControl(right_control);

      if (!status.IsOK()) {
        AOS_LOG(ERROR, "Failed to write control to falcon: %s: %s",
                status.GetName(), status.GetDescription());
      }
    }
  }

  void Stop() override {
    AOS_LOG(WARNING, "drivetrain output too old\n");
    ctre::phoenix6::controls::DutyCycleOut stop_command(0.0);
    stop_command.UpdateFreqHz = 0_Hz;
    stop_command.EnableFOC = true;

    for (auto falcon : {right_front_.get(), right_back_.get(),
                        left_front_.get(), left_back_.get()}) {
      falcon->talon()->SetControl(stop_command);
    }
  }

  double SafeSpeed(double voltage) {
    return (::aos::Clip(voltage, -kMaxBringupPower, kMaxBringupPower) / 12.0);
  }

  ctre::phoenix6::signals::InvertedValue left_inverted_, right_inverted_;
  std::shared_ptr<Falcon> right_front_, right_back_, left_front_, left_back_;
};

class WPILibRobot : public ::frc971::wpilib::WPILibRobotBase {
 public:
  ::std::unique_ptr<frc::Encoder> make_encoder(int index) {
    return make_unique<frc::Encoder>(10 + index * 2, 11 + index * 2, false,
                                     frc::Encoder::k4X);
  }

  void Run() override {
    std::shared_ptr<const Values> values =
        std::make_shared<const Values>(constants::MakeValues());

    aos::FlatbufferDetachedBuffer<aos::Configuration> config =
        aos::configuration::ReadConfig("aos_config.json");

    // Thread 1.
    ::aos::ShmEventLoop joystick_sender_event_loop(&config.message());
    ::frc971::wpilib::JoystickSender joystick_sender(
        &joystick_sender_event_loop);
    AddLoop(&joystick_sender_event_loop);

    // Thread 2.
    ::aos::ShmEventLoop pdp_fetcher_event_loop(&config.message());
    ::frc971::wpilib::PDPFetcher pdp_fetcher(&pdp_fetcher_event_loop);
    AddLoop(&pdp_fetcher_event_loop);

    std::shared_ptr<frc::DigitalOutput> superstructure_reading =
        make_unique<frc::DigitalOutput>(25);

    std::vector<ctre::phoenix6::BaseStatusSignal *> signals_registry;
    std::shared_ptr<Falcon> right_front =
        std::make_shared<Falcon>(1, "Drivetrain Bus", &signals_registry);
    std::shared_ptr<Falcon> right_back =
        std::make_shared<Falcon>(0, "Drivetrain Bus", &signals_registry);
    std::shared_ptr<Falcon> left_front =
        std::make_shared<Falcon>(2, "Drivetrain Bus", &signals_registry);
    std::shared_ptr<Falcon> left_back =
        std::make_shared<Falcon>(3, "Drivetrain Bus", &signals_registry);

    // Thread 3.
    ::aos::ShmEventLoop can_sensor_reader_event_loop(&config.message());
    can_sensor_reader_event_loop.set_name("CANSensorReader");
    CANSensorReader can_sensor_reader(&can_sensor_reader_event_loop,
                                      std::move(signals_registry));

    can_sensor_reader.set_falcons(right_front, right_back, left_front,
                                  left_back);

    AddLoop(&can_sensor_reader_event_loop);

    // Thread 4.
    ::aos::ShmEventLoop sensor_reader_event_loop(&config.message());
    SensorReader sensor_reader(&sensor_reader_event_loop, values,
                               &can_sensor_reader);
    sensor_reader.set_pwm_trigger(true);
    sensor_reader.set_drivetrain_left_encoder(make_encoder(4));
    sensor_reader.set_drivetrain_right_encoder(make_encoder(5));
    sensor_reader.set_superstructure_reading(superstructure_reading);
    sensor_reader.set_yaw_rate_input(make_unique<frc::DigitalInput>(3));

    AddLoop(&sensor_reader_event_loop);

    // Thread 5.
    // Set up CAN.
    if (!FLAGS_ctre_diag_server) {
      c_Phoenix_Diagnostics_SetSecondsToStart(-1);
      c_Phoenix_Diagnostics_Dispose();
    }

    ctre::phoenix::platform::can::CANComm_SetRxSchedPriority(
        constants::Values::kDrivetrainRxPriority, true, "Drivetrain Bus");
    ctre::phoenix::platform::can::CANComm_SetTxSchedPriority(
        constants::Values::kDrivetrainTxPriority, true, "Drivetrain Bus");

    ::aos::ShmEventLoop can_output_event_loop(&config.message());
    can_output_event_loop.set_name("CANOutputWriter");
    DrivetrainWriter drivetrain_writer(&can_output_event_loop);

    drivetrain_writer.set_falcons(right_front, right_back, left_front,
                                  left_back);
    drivetrain_writer.set_right_inverted(
        ctre::phoenix6::signals::InvertedValue::CounterClockwise_Positive);
    drivetrain_writer.set_left_inverted(
        ctre::phoenix6::signals::InvertedValue::Clockwise_Positive);

    can_output_event_loop.MakeWatcher(
        "/roborio",
        [&drivetrain_writer](const frc971::CANConfiguration &configuration) {
          drivetrain_writer.HandleCANConfiguration(configuration);
        });

    AddLoop(&can_output_event_loop);

    RunLoops();
  }
};

}  // namespace wpilib
}  // namespace y2023_bot3

AOS_ROBOT_CLASS(::y2023_bot3::wpilib::WPILibRobot);
