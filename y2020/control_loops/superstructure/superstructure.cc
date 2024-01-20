#include "y2020/control_loops/superstructure/superstructure.h"

#include "aos/containers/sized_array.h"
#include "aos/events/event_loop.h"
#include "aos/network/team_number.h"

namespace y2020::control_loops::superstructure {

using frc971::control_loops::AbsoluteAndAbsoluteEncoderProfiledJointStatus;
using frc971::control_loops::AbsoluteEncoderProfiledJointStatus;
using frc971::control_loops::PotAndAbsoluteEncoderProfiledJointStatus;

Superstructure::Superstructure(::aos::EventLoop *event_loop,
                               const ::std::string &name)
    : frc971::controls::ControlLoop<Goal, Position, Status, Output>(event_loop,
                                                                    name),
      hood_(constants::GetValues().hood),
      intake_joint_(constants::GetValues().intake),
      turret_(constants::GetValues().turret.subsystem_params),
      drivetrain_status_fetcher_(
          event_loop->MakeFetcher<frc971::control_loops::drivetrain::Status>(
              "/drivetrain")),
      joystick_state_fetcher_(
          event_loop->MakeFetcher<aos::JoystickState>("/aos")),
      has_turret_(::aos::network::GetTeamNumber() != 9971) {
  event_loop->SetRuntimeRealtimePriority(30);
}

double Superstructure::robot_speed() const {
  return (drivetrain_status_fetcher_.get() != nullptr
              ? drivetrain_status_fetcher_->robot_speed()
              : 0.0);
}

void Superstructure::RunIteration(const Goal *unsafe_goal,
                                  const Position *position,
                                  aos::Sender<Output>::Builder *output,
                                  aos::Sender<Status>::Builder *status) {
  if (WasReset()) {
    AOS_LOG(ERROR, "WPILib reset, restarting\n");
    hood_.Reset();
    intake_joint_.Reset();
    turret_.Reset();
  }

  const aos::monotonic_clock::time_point position_timestamp =
      event_loop()->context().monotonic_event_time;

  if (drivetrain_status_fetcher_.Fetch()) {
    aos::Alliance alliance = aos::Alliance::kInvalid;
    joystick_state_fetcher_.Fetch();
    if (joystick_state_fetcher_.get() != nullptr) {
      alliance = joystick_state_fetcher_->alliance();
    }
    const turret::Aimer::WrapMode mode =
        (unsafe_goal != nullptr && unsafe_goal->shooting())
            ? turret::Aimer::WrapMode::kAvoidWrapping
            : turret::Aimer::WrapMode::kAvoidEdges;
    aimer_.Update(drivetrain_status_fetcher_.get(), alliance, mode,
                  turret::Aimer::ShotMode::kShootOnTheFly);
  }

  const float velocity = robot_speed();

  const flatbuffers::Offset<AimerStatus> aimer_status_offset =
      aimer_.PopulateStatus(status->fbb());

  const double distance_to_goal = aimer_.DistanceToGoal();

  aos::FlatbufferFixedAllocatorArray<
      frc971::control_loops::StaticZeroingSingleDOFProfiledSubsystemGoal, 64>
      hood_goal;
  aos::FlatbufferFixedAllocatorArray<ShooterGoal, 64> shooter_goal;

  constants::Values::ShotParams shot_params;
  if (constants::GetValues().shot_interpolation_table.GetInRange(
          distance_to_goal, &shot_params)) {
    hood_goal.Finish(frc971::control_loops::
                         CreateStaticZeroingSingleDOFProfiledSubsystemGoal(
                             *hood_goal.fbb(), shot_params.hood_angle));

    shooter_goal.Finish(CreateShooterGoal(*shooter_goal.fbb(),
                                          shot_params.velocity_accelerator,
                                          shot_params.velocity_finisher));
  } else {
    hood_goal.Finish(
        frc971::control_loops::
            CreateStaticZeroingSingleDOFProfiledSubsystemGoal(
                *hood_goal.fbb(), constants::GetValues().hood.range.upper));

    shooter_goal.Finish(CreateShooterGoal(*shooter_goal.fbb(), 0.0, 0.0));
  }

  OutputT output_struct;

  flatbuffers::Offset<AbsoluteAndAbsoluteEncoderProfiledJointStatus>
      hood_status_offset = hood_.Iterate(
          unsafe_goal != nullptr
              ? (unsafe_goal->hood_tracking() ? &hood_goal.message()
                                              : unsafe_goal->hood())
              : nullptr,
          position->hood(),
          output != nullptr ? &(output_struct.hood_voltage) : nullptr,
          status->fbb());

  bool intake_out_jostle = false;

  if (unsafe_goal != nullptr) {
    if (unsafe_goal->shooting() &&
        shooting_start_time_ == aos::monotonic_clock::min_time) {
      shooting_start_time_ = position_timestamp;
    }

    if (unsafe_goal->shooting()) {
      intake_joint_.set_max_acceleration(30.0);
      constexpr std::chrono::milliseconds kPeriod =
          std::chrono::milliseconds(250);
      if ((position_timestamp - shooting_start_time_) % (kPeriod * 2) <
          kPeriod) {
        intake_joint_.set_min_position(-0.25);
        intake_out_jostle = false;
      } else {
        intake_out_jostle = true;
        intake_joint_.set_min_position(-0.75);
      }
    } else {
      intake_joint_.clear_max_acceleration();
      intake_joint_.clear_min_position();
    }

    if (!unsafe_goal->shooting()) {
      shooting_start_time_ = aos::monotonic_clock::min_time;
    }
  }

  flatbuffers::Offset<AbsoluteEncoderProfiledJointStatus> intake_status_offset =
      intake_joint_.Iterate(
          unsafe_goal != nullptr ? unsafe_goal->intake() : nullptr,
          position->intake_joint(),
          output != nullptr ? &(output_struct.intake_joint_voltage) : nullptr,
          status->fbb());

  const frc971::control_loops::StaticZeroingSingleDOFProfiledSubsystemGoal
      *turret_goal = unsafe_goal != nullptr ? (unsafe_goal->turret_tracking()
                                                   ? aimer_.TurretGoal()
                                                   : unsafe_goal->turret())
                                            : nullptr;

  flatbuffers::Offset<PotAndAbsoluteEncoderProfiledJointStatus>
      turret_status_offset;
  if (has_turret_) {
    turret_status_offset = turret_.Iterate(
        turret_goal, position->turret(),
        output != nullptr ? &(output_struct.turret_voltage) : nullptr,
        status->fbb());
  } else {
    PotAndAbsoluteEncoderProfiledJointStatus::Builder turret_builder(
        *status->fbb());
    turret_builder.add_position(M_PI);
    turret_builder.add_velocity(0.0);
    turret_status_offset = turret_builder.Finish();
  }

  flatbuffers::Offset<ShooterStatus> shooter_status_offset =
      shooter_.RunIteration(
          unsafe_goal != nullptr
              ? (unsafe_goal->shooter_tracking() ? &shooter_goal.message()
                                                 : unsafe_goal->shooter())
              : nullptr,
          position->shooter(), status->fbb(),
          output != nullptr ? &(output_struct) : nullptr, position_timestamp);

  const AbsoluteAndAbsoluteEncoderProfiledJointStatus *const hood_status =
      GetMutableTemporaryPointer(*status->fbb(), hood_status_offset);

  const PotAndAbsoluteEncoderProfiledJointStatus *const turret_status =
      GetMutableTemporaryPointer(*status->fbb(), turret_status_offset);

  if (output != nullptr) {
    // Friction is a pain and putting a really high burden on the integrator.
    // TODO(james): I'm not sure how helpful this gain is.
    const double turret_velocity_sign =
        turret_status->velocity() * kTurretFrictionGain;
    output_struct.turret_voltage +=
        std::clamp(turret_velocity_sign, -kTurretFrictionVoltageLimit,
                   kTurretFrictionVoltageLimit);
    const double time_sec =
        aos::time::DurationInSeconds(position_timestamp.time_since_epoch());
    output_struct.turret_voltage +=
        kTurretDitherGain * std::sin(2.0 * M_PI * time_sec * 30.0);
    output_struct.turret_voltage =
        std::clamp(output_struct.turret_voltage, -turret_.operating_voltage(),
                   turret_.operating_voltage());
  }

  bool zeroed;
  bool estopped;

  {
    const AbsoluteEncoderProfiledJointStatus *const intake_status =
        GetMutableTemporaryPointer(*status->fbb(), intake_status_offset);

    zeroed = hood_status->zeroed() && intake_status->zeroed() &&
             turret_status->zeroed();
    estopped = hood_status->estopped() || intake_status->estopped() ||
               turret_status->estopped();
  }

  flatbuffers::Offset<flatbuffers::Vector<Subsystem>>
      subsystems_not_ready_offset;
  const bool turret_ready =
      (std::abs(turret_.goal(0) - turret_.position()) < 0.025) || !has_turret_;
  if (unsafe_goal && unsafe_goal->shooting() &&
      (!shooter_.ready() || !turret_ready)) {
    aos::SizedArray<Subsystem, 3> subsystems_not_ready;
    if (!shooter_.finisher_ready()) {
      subsystems_not_ready.push_back(Subsystem::FINISHER);
    }
    if (!shooter_.accelerator_ready()) {
      subsystems_not_ready.push_back(Subsystem::ACCELERATOR);
    }
    if (!turret_ready) {
      subsystems_not_ready.push_back(Subsystem::TURRET);
    }

    subsystems_not_ready_offset = status->fbb()->CreateVector(
        subsystems_not_ready.data(), subsystems_not_ready.size());
  }

  Status::Builder status_builder = status->MakeBuilder<Status>();

  status_builder.add_zeroed(zeroed);
  status_builder.add_estopped(estopped);

  status_builder.add_hood(hood_status_offset);
  status_builder.add_intake(intake_status_offset);
  status_builder.add_turret(turret_status_offset);
  status_builder.add_shooter(shooter_status_offset);
  status_builder.add_aimer(aimer_status_offset);
  status_builder.add_subsystems_not_ready(subsystems_not_ready_offset);

  status_builder.add_send_failures(status_failure_counter_.failures());

  status_failure_counter_.Count(status->Send(status_builder.Finish()));

  if (output != nullptr) {
    output_struct.washing_machine_spinner_voltage = 0.0;
    output_struct.feeder_voltage = 0.0;
    output_struct.intake_roller_voltage = 0.0;
    output_struct.climber_voltage = 0.0;
    if (unsafe_goal) {
      if (unsafe_goal->has_turret()) {
        output_struct.climber_voltage =
            std::clamp(unsafe_goal->climber_voltage(), -12.0f, 12.0f);

        // Make sure the turret is relatively close to the goal before turning
        // the climber on.
        CHECK(unsafe_goal->has_turret());
        if (std::abs(unsafe_goal->turret()->unsafe_goal() -
                     turret_.position()) > 0.1 &&
            has_turret_) {
          output_struct.climber_voltage = 0;
        }
      }

      if (unsafe_goal->shooting() || unsafe_goal->intake_preloading()) {
        preloading_timeout_ = position_timestamp + kPreloadingTimeout;
      }

      if (position_timestamp <= preloading_timeout_ &&
          !position->intake_beambreak_triggered()) {
        output_struct.washing_machine_spinner_voltage = 5.0;
        output_struct.feeder_voltage = 12.0;

        preloading_backpower_timeout_ =
            position_timestamp + kPreloadingBackpowerDuration;
      }

      if (position->intake_beambreak_triggered() &&
          position_timestamp <= preloading_backpower_timeout_) {
        output_struct.feeder_voltage = -12.0;
      }

      if (unsafe_goal->has_feed_voltage_override()) {
        output_struct.feeder_voltage = unsafe_goal->feed_voltage_override();
        output_struct.washing_machine_spinner_voltage = -5.0;
        preloading_timeout_ = position_timestamp;
      }

      if (unsafe_goal->shooting()) {
        if ((shooter_.ready() ||
             (!has_turret_ && shooter_.accelerator_ready())) &&
            turret_ready) {
          output_struct.feeder_voltage = 12.0;
        }

        if (!intake_out_jostle) {
          output_struct.washing_machine_spinner_voltage = 5.0;
        } else {
          output_struct.washing_machine_spinner_voltage = -5.0;
        }
        output_struct.intake_roller_voltage = 3.0;
      } else {
        output_struct.intake_roller_voltage =
            unsafe_goal->roller_voltage() +
            std::max(velocity * unsafe_goal->roller_speed_compensation(), 0.0f);
      }
    }

    output->CheckOk(output->Send(Output::Pack(*output->fbb(), &output_struct)));
  }
}

}  // namespace y2020::control_loops::superstructure
