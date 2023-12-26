#include "y2023_bot3/autonomous/autonomous_actor.h"

#include <chrono>
#include <cinttypes>
#include <cmath>

#include "aos/logging/logging.h"
#include "aos/util/math.h"
#include "frc971/control_loops/drivetrain/localizer_generated.h"
#include "y2023_bot3/autonomous/auto_splines.h"
#include "y2023_bot3/constants.h"
#include "y2023_bot3/control_loops/drivetrain/drivetrain_base.h"

DEFINE_bool(spline_auto, false, "Run simple test S-spline auto mode.");
DEFINE_bool(charged_up, true,
            "If true run charged up autonomous mode. 2 Piece non-cable side");
DEFINE_bool(charged_up_middle, false,
            "If true run charged up middle autonomous mode. Starts middle, "
            "places cube mid, mobility");
DEFINE_bool(one_piece, false,
            "End charged_up autonomous after first cube is placed.");

namespace y2023_bot3 {
namespace autonomous {

using ::frc971::ProfileParametersT;

ProfileParametersT MakeProfileParameters(float max_velocity,
                                         float max_acceleration) {
  ProfileParametersT result;
  result.max_velocity = max_velocity;
  result.max_acceleration = max_acceleration;
  return result;
}

using ::aos::monotonic_clock;
using frc971::CreateProfileParameters;
using ::frc971::ProfileParametersT;
using frc971::control_loops::CreateStaticZeroingSingleDOFProfiledSubsystemGoal;
using frc971::control_loops::StaticZeroingSingleDOFProfiledSubsystemGoal;
using frc971::control_loops::drivetrain::LocalizerControl;
namespace chrono = ::std::chrono;

AutonomousActor::AutonomousActor(::aos::EventLoop *event_loop)
    : frc971::autonomous::BaseAutonomousActor(
          event_loop, control_loops::drivetrain::GetDrivetrainConfig()),
      localizer_control_sender_(
          event_loop->MakeSender<
              ::frc971::control_loops::drivetrain::LocalizerControl>(
              "/drivetrain")),
      joystick_state_fetcher_(
          event_loop->MakeFetcher<aos::JoystickState>("/aos")),
      robot_state_fetcher_(event_loop->MakeFetcher<aos::RobotState>("/aos")),
      auto_splines_(),
      superstructure_goal_sender_(
          event_loop
              ->MakeSender<::y2023_bot3::control_loops::superstructure::Goal>(
                  "/superstructure")),
      superstructure_status_fetcher_(
          event_loop->MakeFetcher<
              ::y2023_bot3::control_loops::superstructure::Status>(
              "/superstructure")) {
  drivetrain_status_fetcher_.Fetch();
  replan_timer_ = event_loop->AddTimer([this]() { Replan(); });

  event_loop->OnRun([this, event_loop]() {
    replan_timer_->Schedule(event_loop->monotonic_now());
    button_poll_->Schedule(event_loop->monotonic_now(),
                           chrono::milliseconds(50));
  });

  // TODO(james): Really need to refactor this code since we keep using it.
  button_poll_ = event_loop->AddTimer([this]() {
    const aos::monotonic_clock::time_point now =
        this->event_loop()->context().monotonic_event_time;
    if (robot_state_fetcher_.Fetch()) {
      if (robot_state_fetcher_->user_button()) {
        user_indicated_safe_to_reset_ = true;
        MaybeSendStartingPosition();
      }
    }
    if (joystick_state_fetcher_.Fetch()) {
      if (joystick_state_fetcher_->has_alliance() &&
          (joystick_state_fetcher_->alliance() != alliance_)) {
        alliance_ = joystick_state_fetcher_->alliance();
        is_planned_ = false;
        // Only kick the planning out by 2 seconds. If we end up enabled in
        // that second, then we will kick it out further based on the code
        // below.
        replan_timer_->Schedule(now + std::chrono::seconds(2));
      }
      if (joystick_state_fetcher_->enabled()) {
        if (!is_planned_) {
          // Only replan once we've been disabled for 5 seconds.
          replan_timer_->Schedule(now + std::chrono::seconds(5));
        }
      }
    }
  });
}

void AutonomousActor::Replan() {
  if (!drivetrain_status_fetcher_.Fetch()) {
    replan_timer_->Schedule(event_loop()->monotonic_now() + chrono::seconds(1));
    AOS_LOG(INFO, "Drivetrain not up, replanning in 1 second");
    return;
  }

  if (alliance_ == aos::Alliance::kInvalid) {
    return;
  }
  sent_starting_position_ = false;
  if (FLAGS_spline_auto) {
    test_spline_ =
        PlanSpline(std::bind(&AutonomousSplines::TestSpline, &auto_splines_,
                             std::placeholders::_1, alliance_),
                   SplineDirection::kForward);

    starting_position_ = test_spline_->starting_position();
  } else if (FLAGS_charged_up) {
    charged_up_splines_ = {
        PlanSpline(std::bind(&AutonomousSplines::Spline1, &auto_splines_,
                             std::placeholders::_1, alliance_),
                   SplineDirection::kBackward),
        PlanSpline(std::bind(&AutonomousSplines::Spline2, &auto_splines_,
                             std::placeholders::_1, alliance_),
                   SplineDirection::kForward),
        PlanSpline(std::bind(&AutonomousSplines::Spline3, &auto_splines_,
                             std::placeholders::_1, alliance_),
                   SplineDirection::kBackward),
        PlanSpline(std::bind(&AutonomousSplines::Spline4, &auto_splines_,
                             std::placeholders::_1, alliance_),
                   SplineDirection::kForward)};
    starting_position_ = charged_up_splines_.value()[0].starting_position();
    CHECK(starting_position_);
  } else if (FLAGS_charged_up_middle) {
    charged_up_middle_splines_ = {
        PlanSpline(std::bind(&AutonomousSplines::SplineMiddle1, &auto_splines_,
                             std::placeholders::_1, alliance_),
                   SplineDirection::kForward)};
  }
  is_planned_ = true;

  MaybeSendStartingPosition();
}  // namespace autonomous

void AutonomousActor::MaybeSendStartingPosition() {
  if (is_planned_ && user_indicated_safe_to_reset_ &&
      !sent_starting_position_) {
    CHECK(starting_position_);
    SendStartingPosition(starting_position_.value());
  }
}

void AutonomousActor::Reset() {
  InitializeEncoders();
  ResetDrivetrain();

  joystick_state_fetcher_.Fetch();
  CHECK(joystick_state_fetcher_.get() != nullptr)
      << "Expect at least one JoystickState message before running auto...";
  alliance_ = joystick_state_fetcher_->alliance();

  SendSuperstructureGoal();
}

bool AutonomousActor::RunAction(
    const ::frc971::autonomous::AutonomousActionParams *params) {
  Reset();

  AOS_LOG(INFO, "Params are %d\n", params->mode());

  if (!user_indicated_safe_to_reset_) {
    AOS_LOG(WARNING, "Didn't send starting position prior to starting auto.");
    CHECK(starting_position_);
    SendStartingPosition(starting_position_.value());
  }
  // Clear this so that we don't accidentally resend things as soon as we
  // replan later.
  user_indicated_safe_to_reset_ = false;
  is_planned_ = false;
  starting_position_.reset();

  AOS_LOG(INFO, "Params are %d\n", params->mode());
  if (alliance_ == aos::Alliance::kInvalid) {
    AOS_LOG(INFO, "Aborting autonomous due to invalid alliance selection.");
    return false;
  }

  if (FLAGS_charged_up) {
    ChargedUp();
  } else {
    AOS_LOG(INFO, "No autonomous mode selected.");
    return false;
  }

  return true;
}

void AutonomousActor::SendStartingPosition(const Eigen::Vector3d &start) {
  // Set up the starting position for the blue alliance.

  auto builder = localizer_control_sender_.MakeBuilder();

  LocalizerControl::Builder localizer_control_builder =
      builder.MakeBuilder<LocalizerControl>();
  localizer_control_builder.add_x(start(0));
  localizer_control_builder.add_y(start(1));
  localizer_control_builder.add_theta(start(2));
  localizer_control_builder.add_theta_uncertainty(0.00001);
  AOS_LOG(INFO, "User button pressed, x: %f y: %f theta: %f", start(0),
          start(1), start(2));
  if (builder.Send(localizer_control_builder.Finish()) !=
      aos::RawSender::Error::kOk) {
    AOS_LOG(ERROR, "Failed to reset localizer.\n");
  }
}

// Charged Up 2 Game Object Autonomous (non-cable side)
void AutonomousActor::ChargedUp() {
  aos::monotonic_clock::time_point start_time = aos::monotonic_clock::now();

  CHECK(charged_up_splines_);

  auto &splines = *charged_up_splines_;

  // Drive to second cube
  if (!splines[0].WaitForPlan()) {
    return;
  }
  splines[0].Start();

  if (!splines[1].WaitForPlan()) {
    return;
  }
  splines[1].Start();

  if (!splines[1].WaitForSplineDistanceRemaining(0.1)) return;

  // Drive to third cube
  if (!splines[2].WaitForPlan()) {
    return;
  }
  splines[2].Start();

  if (!splines[2].WaitForSplineDistanceRemaining(0.02)) {
    return;
  }

  AOS_LOG(
      INFO, "Got there %lf s\n",
      aos::time::DurationInSeconds(aos::monotonic_clock::now() - start_time));

  // Drive back to grid
  if (!splines[3].WaitForPlan()) {
    return;
  }
  splines[3].Start();
  std::this_thread::sleep_for(chrono::milliseconds(600));

  if (!splines[3].WaitForSplineDistanceRemaining(0.1)) return;
}

// Charged Up Place and Mobility Autonomous (middle)
void AutonomousActor::ChargedUpMiddle() {
  CHECK(charged_up_middle_splines_);

  auto &splines = *charged_up_middle_splines_;

  AOS_LOG(INFO, "Going to preload");

  if (!splines[0].WaitForPlan()) {
    return;
  }
  splines[0].Start();
}

void AutonomousActor::SendSuperstructureGoal() {
  auto builder = superstructure_goal_sender_.MakeBuilder();

  control_loops::superstructure::Goal::Builder superstructure_builder =
      builder.MakeBuilder<control_loops::superstructure::Goal>();

  if (builder.Send(superstructure_builder.Finish()) !=
      aos::RawSender::Error::kOk) {
    AOS_LOG(ERROR, "Sending superstructure goal failed.\n");
  }
}

}  // namespace autonomous
}  // namespace y2023_bot3
