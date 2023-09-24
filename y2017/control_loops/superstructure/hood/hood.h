#ifndef Y2017_CONTROL_LOOPS_SUPERSTRUCTURE_HOOD_HOOD_H_
#define Y2017_CONTROL_LOOPS_SUPERSTRUCTURE_HOOD_HOOD_H_

#include "frc971/control_loops/profiled_subsystem.h"
#include "frc971/zeroing/pulse_index.h"
#include "y2017/constants.h"
#include "y2017/control_loops/superstructure/superstructure_goal_generated.h"

namespace y2017 {
namespace control_loops {
namespace superstructure {
namespace hood {

// Profiled subsystem class with significantly relaxed limits while zeroing.  We
// need relaxed limits, because if you start at the top of the range, you need
// to go to -range, and if you start at the bottom of the range, you need to go
// to +range.  The standard subsystem doesn't support that.
class IndexPulseProfiledSubsystem
    : public ::frc971::control_loops::SingleDOFProfiledSubsystem<
          ::frc971::zeroing::PulseIndexZeroingEstimator> {
 public:
  IndexPulseProfiledSubsystem();

 private:
  void CapGoal(const char *name, Eigen::Matrix<double, 3, 1> *goal,
               bool print) override;
};

class Hood {
 public:
  Hood();
  double goal(int row, int col) const {
    return profiled_subsystem_.goal(row, col);
  }

  // The zeroing and operating voltages.
  static constexpr double kZeroingVoltage = 2.0;
  static constexpr double kOperatingVoltage = 12.0;

  // Constants needed when determining if the hood is in danger of breaking its
  // own gears at the normal operating voltage.
  static constexpr double kErrorOnPositionTillNotMoving =
      2.5 * (::y2017::constants::Values::kHoodEncoderIndexDifference /
             ::y2017::constants::Values::kHoodEncoderCountsPerRevolution);

  static constexpr ::aos::monotonic_clock::duration kTimeTillNotMoving =
      ::std::chrono::milliseconds(15);
  static constexpr double kNotMovingVoltage = 2.0;

  flatbuffers::Offset<frc971::control_loops::IndexProfiledJointStatus> Iterate(
      ::aos::monotonic_clock::time_point monotonic_now,
      const double *unsafe_goal,
      const frc971::ProfileParameters *unsafe_goal_profile_parameters,
      const ::frc971::IndexPosition *position, double *output,
      flatbuffers::FlatBufferBuilder *fbb);

  void Reset();

  enum class State : int32_t {
    UNINITIALIZED,
    DISABLED_INITIALIZED,
    ZEROING,
    RUNNING,
    ESTOP,
  };

  State state() const { return state_; }

 private:
  State state_;

  IndexPulseProfiledSubsystem profiled_subsystem_;
  double last_position_ = 0;
  ::aos::monotonic_clock::time_point last_move_time_ =
      ::aos::monotonic_clock::min_time;
};

}  // namespace hood
}  // namespace superstructure
}  // namespace control_loops
}  // namespace y2017

#endif  // Y2017_CONTROL_LOOPS_SUPERSTRUCTURE_HOOD_HOOD_H_
