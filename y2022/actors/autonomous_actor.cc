#include "y2022/actors/autonomous_actor.h"

#include <chrono>
#include <cinttypes>
#include <cmath>

#include "absl/flags/flag.h"

#include "aos/logging/logging.h"
#include "aos/network/team_number.h"
#include "aos/util/math.h"
#include "frc971/control_loops/drivetrain/localizer_generated.h"
#include "y2022/actors/auto_splines.h"
#include "y2022/constants.h"
#include "y2022/control_loops/drivetrain/drivetrain_base.h"

ABSL_FLAG(bool, spline_auto, false, "If true, define a spline autonomous mode");
ABSL_FLAG(bool, rapid_react, true,
          "If true, run the main rapid react autonomous mode");
ABSL_FLAG(bool, rapid_react_two, false,
          "If true, run the two ball rapid react autonomous mode");

namespace y2022::actors {
namespace {
constexpr double kExtendIntakeGoal = -0.10;
constexpr double kRetractIntakeGoal = 1.47;
constexpr double kIntakeRollerVoltage = 12.0;
constexpr double kRollerVoltage = 12.0;
constexpr double kCatapultReturnPosition = -0.908;
}  // namespace

using ::aos::monotonic_clock;
using frc971::CreateProfileParameters;
using ::frc971::ProfileParametersT;
using frc971::control_loops::CreateStaticZeroingSingleDOFProfiledSubsystemGoal;
using frc971::control_loops::StaticZeroingSingleDOFProfiledSubsystemGoal;
using frc971::control_loops::catapult::CatapultGoal;
using frc971::control_loops::drivetrain::LocalizerControl;

namespace chrono = ::std::chrono;

AutonomousActor::AutonomousActor(::aos::EventLoop *event_loop)
    : frc971::autonomous::BaseAutonomousActor(
          event_loop, control_loops::drivetrain::GetDrivetrainConfig()),
      localizer_control_sender_(
          event_loop->MakeSender<
              ::frc971::control_loops::drivetrain::LocalizerControl>(
              "/drivetrain")),
      superstructure_goal_sender_(
          event_loop->MakeSender<control_loops::superstructure::Goal>(
              "/superstructure")),
      superstructure_status_fetcher_(
          event_loop->MakeFetcher<control_loops::superstructure::Status>(
              "/superstructure")),
      joystick_state_fetcher_(
          event_loop->MakeFetcher<aos::JoystickState>("/aos")),
      robot_state_fetcher_(event_loop->MakeFetcher<aos::RobotState>("/aos")),
      auto_splines_() {
  set_max_drivetrain_voltage(12.0);
  replan_timer_ = event_loop->AddTimer([this]() { Replan(); });
  event_loop->OnRun([this, event_loop]() {
    replan_timer_->Schedule(event_loop->monotonic_now());
    button_poll_->Schedule(event_loop->monotonic_now(),
                           chrono::milliseconds(50));
  });

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
        // Only kick the planning out by 2 seconds. If we end up enabled in that
        // second, then we will kick it out further based on the code below.
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
  LOG(INFO) << "Alliance " << static_cast<int>(alliance_);
  if (alliance_ == aos::Alliance::kInvalid) {
    return;
  }
  sent_starting_position_ = false;
  if (absl::GetFlag(FLAGS_spline_auto)) {
    test_spline_ =
        PlanSpline(std::bind(&AutonomousSplines::TestSpline, &auto_splines_,
                             std::placeholders::_1, alliance_),
                   SplineDirection::kForward);

    starting_position_ = test_spline_->starting_position();
  } else if (absl::GetFlag(FLAGS_rapid_react)) {
    rapid_react_splines_ = {
        PlanSpline(std::bind(&AutonomousSplines::Spline1, &auto_splines_,
                             std::placeholders::_1, alliance_),
                   SplineDirection::kBackward),
        PlanSpline(std::bind(&AutonomousSplines::Spline2, &auto_splines_,
                             std::placeholders::_1, alliance_),
                   SplineDirection::kBackward),
        PlanSpline(std::bind(&AutonomousSplines::Spline3, &auto_splines_,
                             std::placeholders::_1, alliance_),
                   SplineDirection::kForward)};
    starting_position_ = rapid_react_splines_.value()[0].starting_position();
    CHECK(starting_position_);
  } else if (absl::GetFlag(FLAGS_rapid_react_two)) {
    rapid_react_two_spline_ = {
        PlanSpline(std::bind(&AutonomousSplines::SplineTwoBall1, &auto_splines_,
                             std::placeholders::_1, alliance_),
                   SplineDirection::kBackward),
        PlanSpline(std::bind(&AutonomousSplines::SplineTwoBall2, &auto_splines_,
                             std::placeholders::_1, alliance_),
                   SplineDirection::kForward)};
    starting_position_ = rapid_react_two_spline_.value()[0].starting_position();
    CHECK(starting_position_);
  }

  is_planned_ = true;

  MaybeSendStartingPosition();
}

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
  RetractFrontIntake();
  RetractBackIntake();

  joystick_state_fetcher_.Fetch();
  CHECK(joystick_state_fetcher_.get() != nullptr)
      << "Expect at least one JoystickState message before running auto...";
  alliance_ = joystick_state_fetcher_->alliance();
}

bool AutonomousActor::RunAction(
    const ::frc971::autonomous::AutonomousActionParams *params) {
  Reset();
  if (!user_indicated_safe_to_reset_) {
    AOS_LOG(WARNING, "Didn't send starting position prior to starting auto.");
    CHECK(starting_position_);
    SendStartingPosition(starting_position_.value());
  }
  // Clear this so that we don't accidentally resend things as soon as we replan
  // later.
  user_indicated_safe_to_reset_ = false;
  is_planned_ = false;
  starting_position_.reset();

  AOS_LOG(INFO, "Params are %d\n", params->mode());
  if (alliance_ == aos::Alliance::kInvalid) {
    AOS_LOG(INFO, "Aborting autonomous due to invalid alliance selection.");
    return false;
  }
  if (absl::GetFlag(FLAGS_spline_auto)) {
    SplineAuto();
  } else if (absl::GetFlag(FLAGS_rapid_react)) {
    RapidReact();
  } else if (absl::GetFlag(FLAGS_rapid_react_two)) {
    RapidReactTwo();
  }

  return true;
}

void AutonomousActor::SendStartingPosition(const Eigen::Vector3d &start) {
  // Set up the starting position for the blue alliance.

  // TODO(james): Resetting the localizer breaks the left/right statespace
  // controller.  That is a bug, but we can fix that later by not resetting.
  auto builder = localizer_control_sender_.MakeBuilder();

  LocalizerControl::Builder localizer_control_builder =
      builder.MakeBuilder<LocalizerControl>();
  localizer_control_builder.add_x(start(0));
  localizer_control_builder.add_y(start(1));
  localizer_control_builder.add_theta(start(2));
  localizer_control_builder.add_theta_uncertainty(0.00001);
  LOG(INFO) << "User button pressed, x: " << start(0) << " y: " << start(1)
            << " theta: " << start(2);
  if (builder.Send(localizer_control_builder.Finish()) !=
      aos::RawSender::Error::kOk) {
    AOS_LOG(ERROR, "Failed to reset localizer.\n");
  }
}

void AutonomousActor::SplineAuto() {
  CHECK(test_spline_);

  if (!test_spline_->WaitForPlan()) return;
  test_spline_->Start();

  if (!test_spline_->WaitForSplineDistanceRemaining(0.02)) return;
}

void AutonomousActor::RapidReact() {
  aos::monotonic_clock::time_point start_time = aos::monotonic_clock::now();

  CHECK(rapid_react_splines_);

  auto &splines = *rapid_react_splines_;

  // Tell the superstructure a ball was preloaded
  if (!WaitForPreloaded()) return;

  // Fire preloaded ball while driving
  set_fire_at_will(true);
  SendSuperstructureGoal();
  if (!WaitForBallsShot()) return;
  LOG(INFO) << "Shot first ball "
            << chrono::duration<double>(aos::monotonic_clock::now() -
                                        start_time)
                   .count()
            << 's';
  set_fire_at_will(false);
  SendSuperstructureGoal();

  // Drive and intake the ball nearest to the starting zone.
  // Fire while moving.
  ExtendBackIntake();
  if (!splines[0].WaitForPlan()) return;
  splines[0].Start();
  // Distance before we don't shoot while moving.
  if (!splines[0].WaitForSplineDistanceRemaining(2.1)) return;
  LOG(INFO) << "Tring to take the shot";

  set_fire_at_will(true);
  SendSuperstructureGoal();

  if (!splines[0].WaitForSplineDistanceRemaining(0.02)) return;

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Fire the last ball we picked up when stopped.
  SendSuperstructureGoal();
  LOG(INFO) << "Close";
  if (!WaitForBallsShot()) return;
  LOG(INFO) << "Shot first 3 balls "
            << chrono::duration<double>(aos::monotonic_clock::now() -
                                        start_time)
                   .count()
            << 's';

  // Drive to the human player station while intaking two balls.
  // Once is already placed down,
  // and one will be rolled to the robot by the human player
  if (!splines[1].WaitForPlan()) return;
  splines[1].Start();

  std::this_thread::sleep_for(std::chrono::milliseconds(1500));

  set_fire_at_will(false);
  SendSuperstructureGoal();

  if (!splines[1].WaitForSplineDistanceRemaining(0.02)) return;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  LOG(INFO) << "At balls 4/5 "
            << chrono::duration<double>(aos::monotonic_clock::now() -
                                        start_time)
                   .count()
            << 's';

  // Drive to the shooting position
  if (!splines[2].WaitForPlan()) return;
  splines[2].Start();
  if (!splines[2].WaitForSplineDistanceRemaining(2.00)) return;
  RetractFrontIntake();

  if (!splines[2].WaitForSplineDistanceRemaining(0.02)) return;
  LOG(INFO) << "Shooting last balls "
            << chrono::duration<double>(aos::monotonic_clock::now() -
                                        start_time)
                   .count()
            << 's';

  // Fire the two balls once we stopped
  set_fire_at_will(true);
  SendSuperstructureGoal();
  if (!WaitForBallsShot()) return;

  LOG(INFO) << "Took "
            << chrono::duration<double>(aos::monotonic_clock::now() -
                                        start_time)
                   .count()
            << 's';
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  set_fire_at_will(false);
  SendSuperstructureGoal();
}

// Rapid React Two Ball Autonomous.
void AutonomousActor::RapidReactTwo() {
  aos::monotonic_clock::time_point start_time = aos::monotonic_clock::now();

  CHECK(rapid_react_two_spline_);

  auto &splines = *rapid_react_two_spline_;

  // Tell the superstructure a ball was preloaded
  if (!WaitForPreloaded()) return;
  set_fire_at_will(true);
  SendSuperstructureGoal();
  if (!WaitForBallsShot()) return;
  LOG(INFO) << "Shot first ball "
            << chrono::duration<double>(aos::monotonic_clock::now() -
                                        start_time)
                   .count()
            << 's';
  set_fire_at_will(false);
  SendSuperstructureGoal();

  ExtendBackIntake();
  if (!splines[0].WaitForPlan()) return;
  splines[0].Start();
  if (!splines[0].WaitForSplineDistanceRemaining(0.02)) return;

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  if (!splines[1].WaitForPlan()) return;
  splines[1].Start();
  if (!splines[1].WaitForSplineDistanceRemaining(0.02)) return;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Fire the ball once we stopped
  set_fire_at_will(true);
  SendSuperstructureGoal();
  if (!WaitForBallsShot()) return;
  LOG(INFO) << "Shot last ball "
            << chrono::duration<double>(aos::monotonic_clock::now() -
                                        start_time)
                   .count()
            << 's';
  set_fire_at_will(false);
  RetractBackIntake();
  SendSuperstructureGoal();
}

[[nodiscard]] bool AutonomousActor::WaitForPreloaded() {
  set_preloaded(true);
  SendSuperstructureGoal();

  ::aos::time::PhasedLoop phased_loop(frc971::controls::kLoopFrequency,
                                      event_loop()->monotonic_now(),
                                      aos::common::actions::kLoopOffset);

  bool loaded = false;
  while (!loaded) {
    if (ShouldCancel()) {
      return false;
    }

    phased_loop.SleepUntilNext();
    superstructure_status_fetcher_.Fetch();
    CHECK(superstructure_status_fetcher_.get() != nullptr);

    loaded = (superstructure_status_fetcher_->state() ==
              control_loops::superstructure::SuperstructureState::LOADED);
  }

  set_preloaded(false);
  SendSuperstructureGoal();

  return true;
}

void AutonomousActor::SendSuperstructureGoal() {
  auto builder = superstructure_goal_sender_.MakeBuilder();

  flatbuffers::Offset<StaticZeroingSingleDOFProfiledSubsystemGoal>
      intake_front_offset = CreateStaticZeroingSingleDOFProfiledSubsystemGoal(
          *builder.fbb(), intake_front_goal_,
          CreateProfileParameters(*builder.fbb(), 20.0, 60.0));

  flatbuffers::Offset<StaticZeroingSingleDOFProfiledSubsystemGoal>
      intake_back_offset = CreateStaticZeroingSingleDOFProfiledSubsystemGoal(
          *builder.fbb(), intake_back_goal_,
          CreateProfileParameters(*builder.fbb(), 20.0, 60.0));

  flatbuffers::Offset<StaticZeroingSingleDOFProfiledSubsystemGoal>
      catapult_return_position_offset =
          CreateStaticZeroingSingleDOFProfiledSubsystemGoal(
              *builder.fbb(), kCatapultReturnPosition,
              CreateProfileParameters(*builder.fbb(), 9.0, 50.0));

  CatapultGoal::Builder catapult_goal_builder(*builder.fbb());
  catapult_goal_builder.add_shot_position(0.03);
  catapult_goal_builder.add_shot_velocity(18.0);
  catapult_goal_builder.add_return_position(catapult_return_position_offset);
  flatbuffers::Offset<CatapultGoal> catapult_goal_offset =
      catapult_goal_builder.Finish();

  superstructure::Goal::Builder superstructure_builder =
      builder.MakeBuilder<superstructure::Goal>();

  superstructure_builder.add_intake_front(intake_front_offset);
  superstructure_builder.add_intake_back(intake_back_offset);
  superstructure_builder.add_roller_speed_compensation(0.0);
  superstructure_builder.add_roller_speed_front(roller_front_voltage_);
  superstructure_builder.add_roller_speed_back(roller_back_voltage_);
  if (requested_intake_.has_value()) {
    superstructure_builder.add_turret_intake(*requested_intake_);
  }
  superstructure_builder.add_transfer_roller_speed(transfer_roller_voltage_);
  superstructure_builder.add_catapult(catapult_goal_offset);
  superstructure_builder.add_fire(fire_);
  superstructure_builder.add_preloaded(preloaded_);
  superstructure_builder.add_auto_aim(true);

  if (builder.Send(superstructure_builder.Finish()) !=
      aos::RawSender::Error::kOk) {
    AOS_LOG(ERROR, "Sending superstructure goal failed.\n");
  }
}

void AutonomousActor::ExtendFrontIntake() {
  set_requested_intake(RequestedIntake::kFront);
  set_intake_front_goal(kExtendIntakeGoal);
  set_roller_front_voltage(kIntakeRollerVoltage);
  set_transfer_roller_voltage(kRollerVoltage);
  SendSuperstructureGoal();
}

void AutonomousActor::RetractFrontIntake() {
  set_requested_intake(std::nullopt);
  set_intake_front_goal(kRetractIntakeGoal);
  set_roller_front_voltage(0.0);
  set_transfer_roller_voltage(0.0);
  SendSuperstructureGoal();
}

void AutonomousActor::ExtendBackIntake() {
  set_requested_intake(RequestedIntake::kBack);
  set_intake_back_goal(kExtendIntakeGoal);
  set_roller_back_voltage(kIntakeRollerVoltage);
  set_transfer_roller_voltage(-kRollerVoltage);
  SendSuperstructureGoal();
}

void AutonomousActor::RetractBackIntake() {
  set_requested_intake(std::nullopt);
  set_intake_back_goal(kRetractIntakeGoal);
  set_roller_back_voltage(0.0);
  set_transfer_roller_voltage(0.0);
  SendSuperstructureGoal();
}

[[nodiscard]] bool AutonomousActor::WaitForBallsShot() {
  superstructure_status_fetcher_.Fetch();
  CHECK(superstructure_status_fetcher_.get());

  ::aos::time::PhasedLoop phased_loop(frc971::controls::kLoopFrequency,
                                      event_loop()->monotonic_now(),
                                      aos::common::actions::kLoopOffset);
  superstructure_status_fetcher_.Fetch();
  CHECK(superstructure_status_fetcher_.get() != nullptr);

  while (true) {
    if (ShouldCancel()) {
      return false;
    }
    phased_loop.SleepUntilNext();
    superstructure_status_fetcher_.Fetch();
    CHECK(superstructure_status_fetcher_.get() != nullptr);

    if (!superstructure_status_fetcher_->front_intake_has_ball() &&
        !superstructure_status_fetcher_->back_intake_has_ball() &&
        superstructure_status_fetcher_->state() ==
            control_loops::superstructure::SuperstructureState::IDLE) {
      return true;
    }
  }
}

}  // namespace y2022::actors
