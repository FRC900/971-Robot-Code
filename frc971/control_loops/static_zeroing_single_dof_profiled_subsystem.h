#ifndef FRC971_CONTROL_LOOPS_STATIC_ZEROING_SINGLE_DOF_PROFILED_SUBSYSTEM_H_
#define FRC971_CONTROL_LOOPS_STATIC_ZEROING_SINGLE_DOF_PROFILED_SUBSYSTEM_H_

#include "frc971/control_loops/profiled_subsystem.h"

namespace frc971 {
namespace control_loops {

// TODO(austin): Use ProfileParametersT...
struct ProfileParametersStruct {
  float max_velocity;
  float max_acceleration;
};

template <typename ZeroingEstimator>
struct StaticZeroingSingleDOFProfiledSubsystemParams {
  // Maximum voltage while the subsystem is zeroing
  double zeroing_voltage;

  // Maximum voltage while the subsystem is running
  double operating_voltage;

  // Maximum velocity (units/s) and acceleration while State::ZEROING
  ProfileParametersStruct zeroing_profile_params;

  // Maximum velocity (units/s) and acceleration while State::RUNNING if max
  // velocity or acceleration in goal profile_params is 0
  ProfileParametersStruct default_profile_params;

  // Maximum range of the subsystem in meters
  ::frc971::constants::Range range;

  // Zeroing constants for PotAndABsoluteEncoder estimator
  typename ZeroingEstimator::ZeroingConstants zeroing_constants;

  // Function that makes the integral loop for the subsystem
  ::std::function<StateFeedbackLoop<3, 1, 1>()> make_integral_loop;
};

// Class for controlling and motion profiling a single degree of freedom
// subsystem with a zeroing strategy of not moving.
template <typename TZeroingEstimator, typename TProfiledJointStatus,
          typename TSubsystemParams = TZeroingEstimator>
class StaticZeroingSingleDOFProfiledSubsystem {
 public:
  StaticZeroingSingleDOFProfiledSubsystem(
      const StaticZeroingSingleDOFProfiledSubsystemParams<TSubsystemParams>
          &params);

  using ZeroingEstimator = TZeroingEstimator;
  using ProfiledJointStatus = TProfiledJointStatus;

  // Returns the filtered goal of the profiled subsystem (R)
  double goal(int row) const { return profiled_subsystem_.goal(row, 0); }

  // Returns the zeroing voltage of the subsystem
  double zeroing_voltage() { return params_.zeroing_voltage; }

  // Returns the operating voltage of the subsystem
  double operating_voltage() { return params_.operating_voltage; }

  // Sets further constraints on the range constant
  void set_min_position(double min_position) { min_position_ = min_position; }

  void set_max_position(double max_position) { max_position_ = max_position; }

  // Sets a temporary acceleration limit.  No accelerations faster than this may
  // be commanded.
  void set_max_acceleration(double max_acceleration) {
    max_acceleration_ = max_acceleration;
  }
  // Clears the acceleration limit.
  void clear_max_acceleration() {
    max_acceleration_ = std::numeric_limits<float>::infinity();
  }

  // Resets constrained min/max position
  void clear_min_position() { min_position_ = params_.range.lower_hard; }

  void clear_max_position() { max_position_ = params_.range.upper_hard; }

  // Returns the current position
  double position() const { return profiled_subsystem_.position(); }

  flatbuffers::Offset<ProfiledJointStatus> Iterate(
      const StaticZeroingSingleDOFProfiledSubsystemGoal *goal,
      const typename ZeroingEstimator::Position *position, double *output,
      flatbuffers::FlatBufferBuilder *status_fbb);

  // Resets the profiled subsystem and returns to uninitialized
  void Reset();

  void TriggerEstimatorError() { profiled_subsystem_.TriggerEstimatorError(); }

  void set_controller_index(int index) {
    profiled_subsystem_.set_controller_index(index);
  }

  enum class State : int32_t {
    UNINITIALIZED,
    DISABLED_INITIALIZED,
    ZEROING,
    RUNNING,
    ESTOP,
  };

  bool zeroed() const { return profiled_subsystem_.zeroed(); }
  bool estopped() const { return state() == State::ESTOP; }

  State state() const { return state_; }

 private:
  State state_ = State::UNINITIALIZED;
  double min_position_, max_position_;
  float max_acceleration_ = std::numeric_limits<float>::infinity();

  const StaticZeroingSingleDOFProfiledSubsystemParams<TSubsystemParams> params_;

  ::frc971::control_loops::SingleDOFProfiledSubsystem<ZeroingEstimator>
      profiled_subsystem_;
};

template <typename ZeroingEstimator, typename ProfiledJointStatus,
          typename SubsystemParams>
StaticZeroingSingleDOFProfiledSubsystem<ZeroingEstimator, ProfiledJointStatus,
                                        SubsystemParams>::
    StaticZeroingSingleDOFProfiledSubsystem(
        const StaticZeroingSingleDOFProfiledSubsystemParams<SubsystemParams>
            &params)
    : params_(params),
      profiled_subsystem_(
          ::std::unique_ptr<
              ::frc971::control_loops::SimpleCappedStateFeedbackLoop<3, 1, 1>>(
              new ::frc971::control_loops::SimpleCappedStateFeedbackLoop<
                  3, 1, 1>(params_.make_integral_loop())),
          params.zeroing_constants, params.range,
          params.default_profile_params.max_velocity,
          params.default_profile_params.max_acceleration) {
  Reset();
};

template <typename ZeroingEstimator, typename ProfiledJointStatus,
          typename SubsystemParams>
void StaticZeroingSingleDOFProfiledSubsystem<
    ZeroingEstimator, ProfiledJointStatus, SubsystemParams>::Reset() {
  state_ = State::UNINITIALIZED;
  clear_min_position();
  clear_max_position();
  profiled_subsystem_.Reset();
}

template <typename ZeroingEstimator, typename ProfiledJointStatus,
          typename SubsystemParams>
flatbuffers::Offset<ProfiledJointStatus>
StaticZeroingSingleDOFProfiledSubsystem<ZeroingEstimator, ProfiledJointStatus,
                                        SubsystemParams>::
    Iterate(const StaticZeroingSingleDOFProfiledSubsystemGoal *goal,
            const typename ZeroingEstimator::Position *position, double *output,
            flatbuffers::FlatBufferBuilder *status_fbb) {
  CHECK_NOTNULL(position);
  CHECK_NOTNULL(status_fbb);
  bool disabled = output == nullptr;
  profiled_subsystem_.Correct(*position);

  if (profiled_subsystem_.error()) {
    state_ = State::ESTOP;
  }

  switch (state_) {
    case State::UNINITIALIZED:
      if (profiled_subsystem_.initialized()) {
        state_ = State::DISABLED_INITIALIZED;
      }
      disabled = true;
      break;
    case State::DISABLED_INITIALIZED:
      // Wait here until we are either fully zeroed while disabled, or we become
      // enabled.
      if (disabled) {
        if (profiled_subsystem_.zeroed()) {
          state_ = State::RUNNING;
        }
      } else {
        state_ = State::ZEROING;
      }

      // Set the goals to where we are now so when we start back up, we don't
      // jump.
      profiled_subsystem_.ForceGoal(profiled_subsystem_.position());
      // Set up the profile to be the zeroing profile.
      profiled_subsystem_.AdjustProfile(
          params_.zeroing_profile_params.max_velocity,
          params_.zeroing_profile_params.max_acceleration);

      // We are not ready to start doing anything yet.
      disabled = true;
      break;
    case State::ZEROING:
      // Now, zero by actively holding still.
      if (profiled_subsystem_.zeroed()) {
        state_ = State::RUNNING;
      } else if (disabled) {
        state_ = State::DISABLED_INITIALIZED;
      }
      break;

    case State::RUNNING: {
      if (disabled) {
        // Reset the profile to the current position so it starts from here when
        // we get re-enabled.
        profiled_subsystem_.ForceGoal(profiled_subsystem_.position());
      }

      if (goal) {
        if (goal->profile_params()) {
          profiled_subsystem_.AdjustProfile(
              goal->profile_params()->max_velocity(),
              std::min(goal->profile_params()->max_acceleration(),
                       max_acceleration_));
        } else {
          profiled_subsystem_.AdjustProfile(
              profiled_subsystem_.default_velocity(),
              std::min(profiled_subsystem_.default_acceleration(),
                       static_cast<double>(max_acceleration_)));
        }

        double safe_goal = goal->unsafe_goal();
        if (safe_goal < min_position_) {
          VLOG(1) << "Limiting to " << min_position_ << " from " << safe_goal;
          safe_goal = min_position_;
        }
        if (safe_goal > max_position_) {
          VLOG(1) << "Limiting to " << max_position_ << " from " << safe_goal;
          safe_goal = max_position_;
        }
        if (goal->has_ignore_profile()) {
          profiled_subsystem_.set_enable_profile(!goal->ignore_profile());
        }
        profiled_subsystem_.set_unprofiled_goal(safe_goal,
                                                goal->goal_velocity());
      }
    } break;

    case State::ESTOP:
      AOS_LOG(ERROR, "Estop\n");
      disabled = true;
      break;
  }

  // Set the voltage limits.
  const double max_voltage = (state_ == State::RUNNING)
                                 ? params_.operating_voltage
                                 : params_.zeroing_voltage;

  profiled_subsystem_.set_max_voltage({{max_voltage}});

  // Calculate the loops for a cycle.
  profiled_subsystem_.Update(disabled);

  // Write out all the voltages.
  if (output) {
    *output = profiled_subsystem_.voltage();
  }

  typename ProfiledJointStatus::Builder status_builder =
      profiled_subsystem_
          .template BuildStatus<typename ProfiledJointStatus::Builder>(
              status_fbb);

  status_builder.add_estopped(estopped());
  status_builder.add_state(static_cast<int32_t>(state_));
  return status_builder.Finish();
}

}  // namespace control_loops
}  // namespace frc971

#endif  // FRC971_CONTROL_LOOPS_STATIC_ZEROING_SINGLE_DOF_PROFILED_SUBSYSTEM_H_
