#ifndef Y2020_CONTROL_LOOPS_SHOOTER_SHOOTER_H_
#define Y2020_CONTROL_LOOPS_SHOOTER_SHOOTER_H_

#include "frc971/control_loops/control_loop.h"
#include "frc971/control_loops/flywheel/flywheel_controller.h"
#include "frc971/control_loops/state_feedback_loop.h"
#include "y2020/control_loops/superstructure/superstructure_goal_generated.h"
#include "y2020/control_loops/superstructure/superstructure_output_generated.h"
#include "y2020/control_loops/superstructure/superstructure_position_generated.h"
#include "y2020/control_loops/superstructure/superstructure_status_generated.h"

namespace y2020 {
namespace control_loops {
namespace superstructure {
namespace shooter {

// Handles all flywheels together.
class Shooter {
 public:
  static constexpr double kVelocityToleranceFinisher = 3.0;
  static constexpr double kVelocityToleranceAccelerator = 4.0;

  Shooter();

  flatbuffers::Offset<ShooterStatus> RunIteration(
      const ShooterGoal *goal, const ShooterPosition *position,
      flatbuffers::FlatBufferBuilder *fbb, OutputT *output,
      const aos::monotonic_clock::time_point position_timestamp);

  bool ready() const { return finisher_ready() && accelerator_ready(); }
  bool finisher_ready() const { return finisher_ready_; }
  bool accelerator_ready() const { return accelerator_ready_; }

  float finisher_goal() const { return finisher_.goal(); }
  float accelerator_goal() const { return accelerator_left_.goal(); }

 private:
  // Minumum difference between the last local maximum finisher angular velocity
  // and the current finisher angular velocity when we have a ball in the
  // flywheel, in radians/s. This arises because the flywheel slows down when
  // there is a ball in it. We can use this to determine when a ball is in the
  // flywheel and when it gets shot.
  static constexpr double kMinFinisherVelocityDipWithBall = 5.0;

  frc971::control_loops::flywheel::FlywheelController finisher_,
      accelerator_left_, accelerator_right_;

  void UpToSpeed(const ShooterGoal *goal);
  bool finisher_ready_ = false;
  bool accelerator_ready_ = false;

  int balls_shot_ = 0;
  bool finisher_goal_changed_ = false;
  bool ball_in_finisher_ = false;
  // Last local maximum in the finisher angular velocity
  double last_finisher_velocity_max_ = 0.0;
  // True if the finisher's average acceleration over the last dt is positive
  bool finisher_accelerating_ = false;

  DISALLOW_COPY_AND_ASSIGN(Shooter);
};

}  // namespace shooter
}  // namespace superstructure
}  // namespace control_loops
}  // namespace y2020

#endif  // Y2020_CONTROL_LOOPS_SHOOTER_SHOOTER_H_
