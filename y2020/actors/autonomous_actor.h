#ifndef y2020_ACTORS_AUTONOMOUS_ACTOR_H_
#define y2020_ACTORS_AUTONOMOUS_ACTOR_H_

#include "aos/actions/actions.h"
#include "aos/actions/actor.h"
#include "frc971/autonomous/base_autonomous_actor.h"
#include "frc971/control_loops/control_loops_generated.h"
#include "frc971/control_loops/drivetrain/drivetrain_config.h"
#include "frc971/control_loops/drivetrain/localizer_generated.h"
#include "y2020/actors/auto_splines.h"
#include "y2020/control_loops/superstructure/superstructure_goal_generated.h"
#include "y2020/control_loops/superstructure/superstructure_status_generated.h"

namespace y2020::actors {

using frc971::control_loops::StaticZeroingSingleDOFProfiledSubsystemGoal;

namespace superstructure = y2020::control_loops::superstructure;

class AutonomousActor : public frc971::autonomous::BaseAutonomousActor {
 public:
  explicit AutonomousActor(aos::EventLoop *event_loop);

  bool RunAction(
      const frc971::autonomous::AutonomousActionParams *params) override;

 private:
  void Reset();

  void set_intake_goal(double intake_goal) { intake_goal_ = intake_goal; }
  void set_roller(bool roller) {
    if (roller) {
      roller_voltage_ = 6.0;
    } else {
      roller_voltage_ = 0.0;
    }
  }
  void set_roller_voltage(double roller_voltage) {
    roller_voltage_ = roller_voltage;
  }
  void set_shooter_tracking(bool shooter_tracking) {
    shooter_tracking_ = shooter_tracking;
  }
  void set_shooting(bool shooting) { shooting_ = shooting; }
  void set_preloading(bool preloading) { preloading_ = preloading; }

  void SendSuperstructureGoal();
  void ExtendIntake();
  void RetractIntake();
  void SplineAuto();
  void Fender();
  void SendStartingPosition(const Eigen::Vector3d &start);
  void TargetAligned();
  void TargetOffset();
  void JustShoot();
  bool DriveFwd();
  bool WaitForBallsShot(int num_shot);
  int Balls();
  bool WaitUntilAbsoluteBallsShot(int absolute_balls);

  void MaybeSendStartingPosition();

  void Replan();

  bool practice_robot_ = false;

  double intake_goal_ = 0.0;
  double roller_voltage_ = 0.0;
  bool shooting_ = false;
  bool preloading_ = true;
  bool shooter_tracking_ = true;
  const float kRollerSpeedCompensation = 2.0;

  aos::Sender<frc971::control_loops::drivetrain::LocalizerControl>
      localizer_control_sender_;
  aos::Sender<y2020::control_loops::superstructure::Goal>
      superstructure_goal_sender_;
  aos::Fetcher<y2020::control_loops::superstructure::Status>
      superstructure_status_fetcher_;
  aos::Fetcher<aos::JoystickState> joystick_state_fetcher_;
  aos::Fetcher<aos::RobotState> robot_state_fetcher_;

  aos::TimerHandler *replan_timer_;
  aos::TimerHandler *button_poll_;

  std::optional<std::array<SplineHandle, 2>> target_offset_splines_;
  // Max number of splines is 5
  std::optional<std::array<SplineHandle, 3>> target_aligned_splines_;

  std::optional<std::array<SplineHandle, 1>> fender_splines_;

  std::optional<SplineHandle> barrel_spline_;
  std::optional<SplineHandle> slalom_spline_;
  std::optional<SplineHandle> test_spline_;

  aos::Alliance alliance_ = aos::Alliance::kInvalid;
  AutonomousSplines auto_splines_;
  bool user_indicated_safe_to_reset_ = false;
  bool sent_starting_position_ = false;

  bool is_planned_ = false;

  std::optional<Eigen::Vector3d> starting_position_;
};

}  // namespace y2020::actors

#endif  // y2020_ACTORS_AUTONOMOUS_ACTOR_H_
