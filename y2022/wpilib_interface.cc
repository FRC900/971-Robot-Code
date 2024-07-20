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

#include "absl/flags/flag.h"
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

#include "ctre/phoenix6/TalonFX.hpp"

#include "aos/commonmath.h"
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
#include "y2022/constants.h"
#include "y2022/control_loops/superstructure/led_indicator.h"
#include "y2022/control_loops/superstructure/superstructure_can_position_generated.h"
#include "y2022/control_loops/superstructure/superstructure_output_generated.h"
#include "y2022/control_loops/superstructure/superstructure_position_static.h"

using ::aos::monotonic_clock;
using ::y2022::constants::Values;
namespace superstructure = ::y2022::control_loops::superstructure;
namespace chrono = ::std::chrono;
using std::make_unique;

ABSL_FLAG(bool, can_catapult, false,
          "If true, use CAN to control the catapult.");

namespace y2022::wpilib {
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

double climber_pot_translate(double voltage) {
  return voltage * Values::kClimberPotMetersPerVolt();
}

double flipper_arms_pot_translate(double voltage) {
  return voltage * Values::kFlipperArmsPotRadiansPerVolt();
}

double intake_pot_translate(double voltage) {
  return voltage * Values::kIntakePotRadiansPerVolt();
}

double turret_pot_translate(double voltage) {
  return voltage * Values::kTurretPotRadiansPerVolt();
}

constexpr double kMaxFastEncoderPulsesPerSecond =
    std::max({Values::kMaxDrivetrainEncoderPulsesPerSecond(),
              Values::kMaxIntakeEncoderPulsesPerSecond()});
static_assert(kMaxFastEncoderPulsesPerSecond <= 1300000,
              "fast encoders are too fast");
constexpr double kMaxMediumEncoderPulsesPerSecond =
    Values::kMaxTurretEncoderPulsesPerSecond();

static_assert(kMaxMediumEncoderPulsesPerSecond <= 400000,
              "medium encoders are too fast");

double catapult_pot_translate(double voltage) {
  return voltage * Values::kCatapultPotRatio() *
         (3.0 /*turns*/ / 5.0 /*volts*/) * (2 * M_PI /*radians*/);
}

void PrintConfigs(ctre::phoenix6::hardware::TalonFX *talon) {
  ctre::phoenix6::configs::TalonFXConfiguration configuration;
  ctre::phoenix::StatusCode status =
      talon->GetConfigurator().Refresh(configuration);
  if (!status.IsOK()) {
    AOS_LOG(ERROR, "Failed to get falcon configuration: %s: %s",
            status.GetName(), status.GetDescription());
  }
  AOS_LOG(INFO, "configuration: %s", configuration.ToString().c_str());
}

void WriteConfigs(ctre::phoenix6::hardware::TalonFX *talon,
                  double stator_current_limit, double supply_current_limit) {
  ctre::phoenix6::configs::CurrentLimitsConfigs current_limits;
  current_limits.StatorCurrentLimit = stator_current_limit;
  current_limits.StatorCurrentLimitEnable = true;
  current_limits.SupplyCurrentLimit = supply_current_limit;
  current_limits.SupplyCurrentLimitEnable = true;

  ctre::phoenix6::configs::TalonFXConfiguration configuration;
  configuration.CurrentLimits = current_limits;

  ctre::phoenix::StatusCode status =
      talon->GetConfigurator().Apply(configuration);
  if (!status.IsOK()) {
    AOS_LOG(ERROR, "Failed to set falcon configuration: %s: %s",
            status.GetName(), status.GetDescription());
  }

  PrintConfigs(talon);
}

void Disable(ctre::phoenix6::hardware::TalonFX *talon) {
  ctre::phoenix6::controls::DutyCycleOut stop_command(0.0);
  stop_command.UpdateFreqHz = 0_Hz;
  stop_command.EnableFOC = true;

  talon->SetControl(stop_command);
}

}  // namespace

// Class to send position messages with sensor readings to our loops.
class SensorReader : public ::frc971::wpilib::SensorReader {
 public:
  SensorReader(::aos::ShmEventLoop *event_loop,
               std::shared_ptr<const Values> values)
      : ::frc971::wpilib::SensorReader(event_loop),
        values_(std::move(values)),
        auto_mode_sender_(
            event_loop->MakeSender<::frc971::autonomous::AutonomousMode>(
                "/autonomous")),
        superstructure_position_sender_(
            event_loop->MakeSender<superstructure::PositionStatic>(
                "/superstructure")),
        drivetrain_position_sender_(
            event_loop
                ->MakeSender<::frc971::control_loops::drivetrain::Position>(
                    "/drivetrain")),
        gyro_sender_(event_loop->MakeSender<::frc971::sensors::GyroReading>(
            "/drivetrain")) {
    // Set to filter out anything shorter than 1/4 of the minimum pulse width
    // we should ever see.
    UpdateFastEncoderFilterHz(kMaxFastEncoderPulsesPerSecond);
    UpdateMediumEncoderFilterHz(kMaxMediumEncoderPulsesPerSecond);
  }

  void Start() override {
    // TODO(Ravago): Figure out why adding multiple DMA readers results in weird
    // behavior
    // AddToDMA(&imu_heading_reader_);
    AddToDMA(&imu_yaw_rate_reader_);
  }

  // Auto mode switches.
  void set_autonomous_mode(int i, ::std::unique_ptr<frc::DigitalInput> sensor) {
    autonomous_modes_.at(i) = ::std::move(sensor);
  }

  void set_catapult_encoder(::std::unique_ptr<frc::Encoder> encoder) {
    medium_encoder_filter_.Add(encoder.get());
    catapult_encoder_.set_encoder(::std::move(encoder));
  }

  void set_catapult_absolute_pwm(
      ::std::unique_ptr<frc::DigitalInput> absolute_pwm) {
    catapult_encoder_.set_absolute_pwm(::std::move(absolute_pwm));
  }

  void set_catapult_potentiometer(
      ::std::unique_ptr<frc::AnalogInput> potentiometer) {
    catapult_encoder_.set_potentiometer(::std::move(potentiometer));
  }

  void set_heading_input(::std::unique_ptr<frc::DigitalInput> sensor) {
    imu_heading_input_ = ::std::move(sensor);
    imu_heading_reader_.set_input(imu_heading_input_.get());
  }

  void set_yaw_rate_input(::std::unique_ptr<frc::DigitalInput> sensor) {
    imu_yaw_rate_input_ = ::std::move(sensor);
    imu_yaw_rate_reader_.set_input(imu_yaw_rate_input_.get());
  }
  void set_catapult_falcon_1(
      ::std::shared_ptr<ctre::phoenix6::hardware::TalonFX> t1,
      ::std::shared_ptr<ctre::phoenix6::hardware::TalonFX> t2) {
    catapult_falcon_1_can_ = ::std::move(t1);
    catapult_falcon_2_can_ = ::std::move(t2);
  }

  void RunIteration() override {
    superstructure_reading_->Set(true);
    {
      aos::Sender<superstructure::PositionStatic>::StaticBuilder builder =
          superstructure_position_sender_.MakeStaticBuilder();

      CopyPosition(catapult_encoder_, builder->add_catapult(),
                   Values::kCatapultEncoderCountsPerRevolution(),
                   Values::kCatapultEncoderRatio(), catapult_pot_translate,
                   false, values_->catapult.potentiometer_offset);

      CopyPosition(*climber_potentiometer_, builder->add_climber(),
                   climber_pot_translate, false,
                   values_->climber.potentiometer_offset);

      CopyPosition(*flipper_arm_left_potentiometer_,
                   builder->add_flipper_arm_left(), flipper_arms_pot_translate,
                   false, values_->flipper_arm_left.potentiometer_offset);

      CopyPosition(*flipper_arm_right_potentiometer_,
                   builder->add_flipper_arm_right(), flipper_arms_pot_translate,
                   true, values_->flipper_arm_right.potentiometer_offset);

      // Intake
      CopyPosition(intake_encoder_front_, builder->add_intake_front(),
                   Values::kIntakeEncoderCountsPerRevolution(),
                   Values::kIntakeEncoderRatio(), intake_pot_translate, true,
                   values_->intake_front.potentiometer_offset);
      CopyPosition(intake_encoder_back_, builder->add_intake_back(),
                   Values::kIntakeEncoderCountsPerRevolution(),
                   Values::kIntakeEncoderRatio(), intake_pot_translate, true,
                   values_->intake_back.potentiometer_offset);
      CopyPosition(turret_encoder_, builder->add_turret(),
                   Values::kTurretEncoderCountsPerRevolution(),
                   Values::kTurretEncoderRatio(), turret_pot_translate, false,
                   values_->turret.potentiometer_offset);

      builder->set_intake_beambreak_front(intake_beambreak_front_->Get());
      builder->set_intake_beambreak_back(intake_beambreak_back_->Get());
      builder->set_turret_beambreak(turret_beambreak_->Get());
      builder.CheckOk(builder.Send());
    }

    {
      auto builder = drivetrain_position_sender_.MakeBuilder();
      frc971::control_loops::drivetrain::Position::Builder drivetrain_builder =
          builder.MakeBuilder<frc971::control_loops::drivetrain::Position>();
      drivetrain_builder.add_left_encoder(
          constants::Values::DrivetrainEncoderToMeters(
              drivetrain_left_encoder_->GetRaw()));
      drivetrain_builder.add_left_speed(
          drivetrain_velocity_translate(drivetrain_left_encoder_->GetPeriod()));

      drivetrain_builder.add_right_encoder(
          -constants::Values::DrivetrainEncoderToMeters(
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
      double heading_duty_cycle =
          imu_heading_reader_.last_width() * kPWMFrequencyHz;
      double velocity_duty_cycle =
          imu_yaw_rate_reader_.last_width() * kPWMFrequencyHz;

      constexpr double kDutyCycleScale =
          1 / (kScaledRangeHigh - kScaledRangeLow);
      // scale from 0.1 - 0.9 to 0 - 1
      double rescaled_heading_duty_cycle =
          (heading_duty_cycle - kScaledRangeLow) * kDutyCycleScale;
      double rescaled_velocity_duty_cycle =
          (velocity_duty_cycle - kScaledRangeLow) * kDutyCycleScale;

      if (!std::isnan(rescaled_heading_duty_cycle)) {
        gyro_reading_builder.add_angle(rescaled_heading_duty_cycle *
                                       (2.0 * M_PI));
      }
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

  void set_climber_potentiometer(
      ::std::unique_ptr<frc::AnalogInput> potentiometer) {
    climber_potentiometer_ = ::std::move(potentiometer);
  }

  void set_flipper_arm_left_potentiometer(
      ::std::unique_ptr<frc::AnalogInput> potentiometer) {
    flipper_arm_left_potentiometer_ = ::std::move(potentiometer);
  }

  void set_flipper_arm_right_potentiometer(
      ::std::unique_ptr<frc::AnalogInput> potentiometer) {
    flipper_arm_right_potentiometer_ = ::std::move(potentiometer);
  }

  std::shared_ptr<frc::DigitalOutput> superstructure_reading_;

  void set_superstructure_reading(
      std::shared_ptr<frc::DigitalOutput> superstructure_reading) {
    superstructure_reading_ = superstructure_reading;
  }

  void set_intake_encoder_front(::std::unique_ptr<frc::Encoder> encoder) {
    fast_encoder_filter_.Add(encoder.get());
    intake_encoder_front_.set_encoder(::std::move(encoder));
  }

  void set_intake_encoder_back(::std::unique_ptr<frc::Encoder> encoder) {
    fast_encoder_filter_.Add(encoder.get());
    intake_encoder_back_.set_encoder(::std::move(encoder));
  }

  void set_intake_front_absolute_pwm(
      ::std::unique_ptr<frc::DigitalInput> absolute_pwm) {
    intake_encoder_front_.set_absolute_pwm(::std::move(absolute_pwm));
  }

  void set_intake_front_potentiometer(
      ::std::unique_ptr<frc::AnalogInput> potentiometer) {
    intake_encoder_front_.set_potentiometer(::std::move(potentiometer));
  }

  void set_intake_back_absolute_pwm(
      ::std::unique_ptr<frc::DigitalInput> absolute_pwm) {
    intake_encoder_back_.set_absolute_pwm(::std::move(absolute_pwm));
  }

  void set_intake_back_potentiometer(
      ::std::unique_ptr<frc::AnalogInput> potentiometer) {
    intake_encoder_back_.set_potentiometer(::std::move(potentiometer));
  }

  void set_turret_encoder(::std::unique_ptr<frc::Encoder> encoder) {
    medium_encoder_filter_.Add(encoder.get());
    turret_encoder_.set_encoder(::std::move(encoder));
  }

  void set_turret_absolute_pwm(
      ::std::unique_ptr<frc::DigitalInput> absolute_pwm) {
    turret_encoder_.set_absolute_pwm(::std::move(absolute_pwm));
  }

  void set_turret_potentiometer(
      ::std::unique_ptr<frc::AnalogInput> potentiometer) {
    turret_encoder_.set_potentiometer(::std::move(potentiometer));
  }

  void set_intake_beambreak_front(::std::unique_ptr<frc::DigitalInput> sensor) {
    intake_beambreak_front_ = ::std::move(sensor);
  }
  void set_intake_beambreak_back(::std::unique_ptr<frc::DigitalInput> sensor) {
    intake_beambreak_back_ = ::std::move(sensor);
  }
  void set_turret_beambreak(::std::unique_ptr<frc::DigitalInput> sensor) {
    turret_beambreak_ = ::std::move(sensor);
  }

 private:
  std::shared_ptr<const Values> values_;

  aos::Sender<frc971::autonomous::AutonomousMode> auto_mode_sender_;
  aos::Sender<superstructure::PositionStatic> superstructure_position_sender_;
  aos::Sender<frc971::control_loops::drivetrain::Position>
      drivetrain_position_sender_;
  ::aos::Sender<::frc971::sensors::GyroReading> gyro_sender_;

  std::array<std::unique_ptr<frc::DigitalInput>, 2> autonomous_modes_;

  std::unique_ptr<frc::DigitalInput> intake_beambreak_front_,
      intake_beambreak_back_, turret_beambreak_, imu_heading_input_,
      imu_yaw_rate_input_;

  std::unique_ptr<frc::AnalogInput> climber_potentiometer_,
      flipper_arm_right_potentiometer_, flipper_arm_left_potentiometer_;
  frc971::wpilib::AbsoluteEncoderAndPotentiometer intake_encoder_front_,
      intake_encoder_back_, turret_encoder_, catapult_encoder_;

  frc971::wpilib::DMAPulseWidthReader imu_heading_reader_, imu_yaw_rate_reader_;

  ::std::shared_ptr<::ctre::phoenix6::hardware::TalonFX> catapult_falcon_1_can_,
      catapult_falcon_2_can_;
};

class SuperstructureWriter
    : public ::frc971::wpilib::LoopOutputHandler<superstructure::Output> {
 public:
  SuperstructureWriter(aos::EventLoop *event_loop)
      : frc971::wpilib::LoopOutputHandler<superstructure::Output>(
            event_loop, "/superstructure"),
        catapult_reversal_(make_unique<frc::DigitalOutput>(0)) {}

  void set_climber_servo_left(::std::unique_ptr<::frc::Servo> t) {
    climber_servo_left_ = ::std::move(t);
  }
  void set_climber_servo_right(::std::unique_ptr<::frc::Servo> t) {
    climber_servo_right_ = ::std::move(t);
  }

  void set_climber_falcon(std::unique_ptr<frc::TalonFX> t) {
    climber_falcon_ = std::move(t);
  }

  void set_turret_falcon(::std::unique_ptr<::frc::TalonFX> t) {
    turret_falcon_ = ::std::move(t);
  }

  void set_catapult_falcon_1(::std::unique_ptr<::frc::TalonFX> t) {
    catapult_falcon_1_ = ::std::move(t);
  }

  void set_catapult_falcon_1(
      ::std::shared_ptr<::ctre::phoenix6::hardware::TalonFX> t1,
      ::std::shared_ptr<::ctre::phoenix6::hardware::TalonFX> t2) {
    catapult_falcon_1_can_ = ::std::move(t1);
    catapult_falcon_2_can_ = ::std::move(t2);

    for (auto &falcon : {catapult_falcon_1_can_, catapult_falcon_2_can_}) {
      ctre::phoenix6::configs::CurrentLimitsConfigs current_limits;
      current_limits.StatorCurrentLimit =
          Values::kIntakeRollerStatorCurrentLimit();
      current_limits.StatorCurrentLimitEnable = true;
      current_limits.SupplyCurrentLimit =
          Values::kIntakeRollerSupplyCurrentLimit();
      current_limits.SupplyCurrentLimitEnable = true;

      ctre::phoenix6::configs::TalonFXConfiguration configuration;
      configuration.CurrentLimits = current_limits;

      ctre::phoenix::StatusCode status =
          falcon->GetConfigurator().Apply(configuration);
      if (!status.IsOK()) {
        AOS_LOG(ERROR, "Failed to set falcon configuration: %s: %s",
                status.GetName(), status.GetDescription());
      }

      PrintConfigs(falcon.get());

      // TODO(max): Figure out how to migrate these configs to phoenix6
      /*falcon->SetStatusFramePeriod(
          ctre::phoenix::motorcontrol::Status_1_General, 1);
      falcon->SetStatusFramePeriod(
          ctre::phoenix::motorcontrol::Status_Brushless_Current, 50);

      falcon->ConfigOpenloopRamp(0.0);
      falcon->ConfigClosedloopRamp(0.0);
      falcon->ConfigVoltageMeasurementFilter(1);*/
    }
  }

  void set_intake_falcon_front(::std::unique_ptr<frc::TalonFX> t) {
    intake_falcon_front_ = ::std::move(t);
  }

  void set_intake_falcon_back(::std::unique_ptr<frc::TalonFX> t) {
    intake_falcon_back_ = ::std::move(t);
  }

  void set_roller_falcon_front(
      ::std::unique_ptr<::ctre::phoenix6::hardware::TalonFX> t) {
    roller_falcon_front_ = ::std::move(t);
    WriteConfigs(roller_falcon_front_.get(),
                 Values::kIntakeRollerStatorCurrentLimit(),
                 Values::kIntakeRollerSupplyCurrentLimit());
  }

  void set_roller_falcon_back(
      ::std::unique_ptr<::ctre::phoenix6::hardware::TalonFX> t) {
    roller_falcon_back_ = ::std::move(t);
    WriteConfigs(roller_falcon_back_.get(),
                 Values::kIntakeRollerStatorCurrentLimit(),
                 Values::kIntakeRollerSupplyCurrentLimit());
  }

  void set_flipper_arms_falcon(
      ::std::shared_ptr<::ctre::phoenix6::hardware::TalonFX> t) {
    flipper_arms_falcon_ = t;
    WriteConfigs(flipper_arms_falcon_.get(),
                 Values::kFlipperArmSupplyCurrentLimit(),
                 Values::kFlipperArmStatorCurrentLimit());
  }

  ::std::shared_ptr<::ctre::phoenix6::hardware::TalonFX> flipper_arms_falcon() {
    return flipper_arms_falcon_;
  }

  void set_transfer_roller_victor(::std::unique_ptr<::frc::VictorSP> t) {
    transfer_roller_victor_ = ::std::move(t);
  }

  std::shared_ptr<frc::DigitalOutput> superstructure_reading_;

  void set_superstructure_reading(
      std::shared_ptr<frc::DigitalOutput> superstructure_reading) {
    superstructure_reading_ = superstructure_reading;
  }

 private:
  void Stop() override {
    AOS_LOG(WARNING, "Superstructure output too old.\n");
    climber_falcon_->SetDisabled();
    climber_servo_left_->SetRaw(0);
    climber_servo_right_->SetRaw(0);

    Disable(roller_falcon_front_.get());
    Disable(roller_falcon_back_.get());
    Disable(flipper_arms_falcon_.get());

    intake_falcon_front_->SetDisabled();
    intake_falcon_back_->SetDisabled();
    transfer_roller_victor_->SetDisabled();
    if (catapult_falcon_1_) {
      catapult_falcon_1_->SetDisabled();
    }
    if (catapult_falcon_1_can_) {
      Disable(catapult_falcon_1_can_.get());
      Disable(catapult_falcon_2_can_.get());
    }
    turret_falcon_->SetDisabled();
  }

  void Write(const superstructure::Output &output) override {
    WritePwm(-output.climber_voltage(), climber_falcon_.get());
    climber_servo_left_->SetPosition(output.climber_servo_left());
    climber_servo_right_->SetPosition(output.climber_servo_right());

    WritePwm(output.intake_voltage_front(), intake_falcon_front_.get());
    WritePwm(output.intake_voltage_back(), intake_falcon_back_.get());
    WriteCan(output.roller_voltage_front(), roller_falcon_front_.get());
    WriteCan(output.roller_voltage_back(), roller_falcon_back_.get());
    WritePwm(output.transfer_roller_voltage(), transfer_roller_victor_.get());

    WriteCan(-output.flipper_arms_voltage(), flipper_arms_falcon_.get());

    if (catapult_falcon_1_) {
      WritePwm(output.catapult_voltage(), catapult_falcon_1_.get());
      superstructure_reading_->Set(false);
      if (output.catapult_voltage() > 0) {
        catapult_reversal_->Set(true);
      } else {
        catapult_reversal_->Set(false);
      }
    }
    if (catapult_falcon_1_can_) {
      WriteCanCatapult(output.catapult_voltage(), catapult_falcon_1_can_.get());
      WriteCanCatapult(output.catapult_voltage(), catapult_falcon_2_can_.get());
    }

    WritePwm(-output.turret_voltage(), turret_falcon_.get());
  }

  static void WriteCan(const double voltage,
                       ::ctre::phoenix6::hardware::TalonFX *falcon) {
    ctre::phoenix6::controls::DutyCycleOut control(
        std::clamp(voltage, -kMaxBringupPower, kMaxBringupPower) / 12.0);
    control.UpdateFreqHz = 0_Hz;
    control.EnableFOC = true;

    falcon->SetControl(control);
  }
  // We do this to set our UpdateFreqHz higher
  static void WriteCanCatapult(const double voltage,
                               ::ctre::phoenix6::hardware::TalonFX *falcon) {
    ctre::phoenix6::controls::DutyCycleOut control(
        std::clamp(voltage, -kMaxBringupPower, kMaxBringupPower) / 12.0);
    control.UpdateFreqHz = 1000_Hz;
    control.EnableFOC = true;

    falcon->SetControl(control);
  }

  template <typename T>
  static void WritePwm(const double voltage, T *motor) {
    motor->SetSpeed(std::clamp(voltage, -kMaxBringupPower, kMaxBringupPower) /
                    12.0);
  }

  ::std::unique_ptr<frc::TalonFX> intake_falcon_front_, intake_falcon_back_;

  ::std::unique_ptr<::ctre::phoenix6::hardware::TalonFX> roller_falcon_front_,
      roller_falcon_back_;

  ::std::shared_ptr<::ctre::phoenix6::hardware::TalonFX> flipper_arms_falcon_;

  ::std::shared_ptr<::ctre::phoenix6::hardware::TalonFX> catapult_falcon_1_can_,
      catapult_falcon_2_can_;

  ::std::unique_ptr<::frc::TalonFX> turret_falcon_, catapult_falcon_1_,
      climber_falcon_;
  ::std::unique_ptr<::frc::VictorSP> transfer_roller_victor_;

  std::unique_ptr<frc::DigitalOutput> catapult_reversal_;

  ::std::unique_ptr<::frc::Servo> climber_servo_left_, climber_servo_right_;
};

class CANSensorReader {
 public:
  CANSensorReader(aos::EventLoop *event_loop)
      : event_loop_(event_loop),
        can_position_sender_(
            event_loop->MakeSender<superstructure::CANPosition>(
                "/superstructure")) {
    event_loop->SetRuntimeRealtimePriority(16);

    phased_loop_handler_ =
        event_loop_->AddPhasedLoop([this](int) { Loop(); }, kPeriod);
    phased_loop_handler_->set_name("CAN SensorReader Loop");

    event_loop->OnRun([this]() { Loop(); });
  }

  void set_flipper_arms_falcon(
      ::std::shared_ptr<::ctre::phoenix6::hardware::TalonFX> t) {
    flipper_arms_falcon_ = std::move(t);
  }

 private:
  void Loop() {
    auto builder = can_position_sender_.MakeBuilder();
    superstructure::CANPosition::Builder can_position_builder =
        builder.MakeBuilder<superstructure::CANPosition>();
    can_position_builder.add_flipper_arm_integrated_sensor_velocity(
        flipper_arms_falcon_->GetVelocity().GetValue().value() *
        kVelocityConversion);
    builder.CheckOk(builder.Send(can_position_builder.Finish()));
  }

  static constexpr std::chrono::milliseconds kPeriod =
      std::chrono::milliseconds(20);
  // 2048 encoder counts / 100 ms to rad/sec
  static constexpr double kVelocityConversion = (2.0 * M_PI / 2048) * 0.100;
  aos::EventLoop *event_loop_;
  ::aos::PhasedLoopHandler *phased_loop_handler_;

  ::std::shared_ptr<::ctre::phoenix6::hardware::TalonFX> flipper_arms_falcon_;
  aos::Sender<superstructure::CANPosition> can_position_sender_;
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

    // Thread 3.
    ::aos::ShmEventLoop sensor_reader_event_loop(&config.message());
    SensorReader sensor_reader(&sensor_reader_event_loop, values);
    sensor_reader.set_pwm_trigger(true);
    sensor_reader.set_drivetrain_left_encoder(make_encoder(1));
    sensor_reader.set_drivetrain_right_encoder(make_encoder(0));
    sensor_reader.set_superstructure_reading(superstructure_reading);

    sensor_reader.set_intake_encoder_front(make_encoder(3));
    sensor_reader.set_intake_front_absolute_pwm(
        make_unique<frc::DigitalInput>(3));
    sensor_reader.set_intake_front_potentiometer(
        make_unique<frc::AnalogInput>(3));

    sensor_reader.set_intake_encoder_back(make_encoder(4));
    sensor_reader.set_intake_back_absolute_pwm(
        make_unique<frc::DigitalInput>(4));
    sensor_reader.set_intake_back_potentiometer(
        make_unique<frc::AnalogInput>(4));

    sensor_reader.set_turret_encoder(make_encoder(5));
    sensor_reader.set_turret_absolute_pwm(make_unique<frc::DigitalInput>(5));
    sensor_reader.set_turret_potentiometer(make_unique<frc::AnalogInput>(5));

    // TODO(milind): correct intake beambreak ports once set
    sensor_reader.set_intake_beambreak_front(make_unique<frc::DigitalInput>(1));
    sensor_reader.set_intake_beambreak_back(make_unique<frc::DigitalInput>(6));
    sensor_reader.set_turret_beambreak(make_unique<frc::DigitalInput>(7));

    sensor_reader.set_climber_potentiometer(make_unique<frc::AnalogInput>(7));

    sensor_reader.set_flipper_arm_left_potentiometer(
        make_unique<frc::AnalogInput>(0));
    sensor_reader.set_flipper_arm_right_potentiometer(
        make_unique<frc::AnalogInput>(1));

    // TODO(milind): correct catapult encoder and absolute pwm ports
    sensor_reader.set_catapult_encoder(make_encoder(2));
    sensor_reader.set_catapult_absolute_pwm(
        std::make_unique<frc::DigitalInput>(2));
    sensor_reader.set_catapult_potentiometer(
        std::make_unique<frc::AnalogInput>(2));

    sensor_reader.set_heading_input(make_unique<frc::DigitalInput>(9));
    sensor_reader.set_yaw_rate_input(make_unique<frc::DigitalInput>(8));

    AddLoop(&sensor_reader_event_loop);

    // Thread 4.
    ::aos::ShmEventLoop output_event_loop(&config.message());
    ::frc971::wpilib::DrivetrainWriter drivetrain_writer(&output_event_loop);
    drivetrain_writer.set_left_controller0(
        ::std::unique_ptr<::frc::VictorSP>(new ::frc::VictorSP(0)), false);
    drivetrain_writer.set_right_controller0(
        ::std::unique_ptr<::frc::VictorSP>(new ::frc::VictorSP(1)), true);

    SuperstructureWriter superstructure_writer(&output_event_loop);

    superstructure_writer.set_turret_falcon(make_unique<::frc::TalonFX>(3));
    superstructure_writer.set_roller_falcon_front(
        make_unique<::ctre::phoenix6::hardware::TalonFX>(0));
    superstructure_writer.set_roller_falcon_back(
        make_unique<::ctre::phoenix6::hardware::TalonFX>(1));

    superstructure_writer.set_transfer_roller_victor(
        make_unique<::frc::VictorSP>(5));

    superstructure_writer.set_intake_falcon_front(make_unique<frc::TalonFX>(2));
    superstructure_writer.set_intake_falcon_back(make_unique<frc::TalonFX>(4));
    superstructure_writer.set_climber_falcon(make_unique<frc::TalonFX>(8));
    superstructure_writer.set_climber_servo_left(make_unique<frc::Servo>(7));
    superstructure_writer.set_climber_servo_right(make_unique<frc::Servo>(6));
    superstructure_writer.set_flipper_arms_falcon(
        make_unique<::ctre::phoenix6::hardware::TalonFX>(2));
    superstructure_writer.set_superstructure_reading(superstructure_reading);

    if (!absl::GetFlag(FLAGS_can_catapult)) {
      superstructure_writer.set_catapult_falcon_1(make_unique<frc::TalonFX>(9));
    } else {
      std::shared_ptr<::ctre::phoenix6::hardware::TalonFX> catapult1 =
          make_unique<::ctre::phoenix6::hardware::TalonFX>(3, "Catapult");
      std::shared_ptr<::ctre::phoenix6::hardware::TalonFX> catapult2 =
          make_unique<::ctre::phoenix6::hardware::TalonFX>(4, "Catapult");
      superstructure_writer.set_catapult_falcon_1(catapult1, catapult2);
      sensor_reader.set_catapult_falcon_1(catapult1, catapult2);
    }

    AddLoop(&output_event_loop);

    // Thread 5.
    ::aos::ShmEventLoop led_indicator_event_loop(&config.message());
    control_loops::superstructure::LedIndicator led_indicator(
        &led_indicator_event_loop);
    AddLoop(&led_indicator_event_loop);

    RunLoops();
  }
};

}  // namespace y2022::wpilib

AOS_ROBOT_CLASS(::y2022::wpilib::WPILibRobot);
