#ifndef Y2021_BOT3_ACTORS_AUTONOMOUS_ACTOR_H_
#define Y2021_BOT3_ACTORS_AUTONOMOUS_ACTOR_H_

#include "aos/actions/actions.h"
#include "aos/actions/actor.h"
#include "frc971/autonomous/base_autonomous_actor.h"
#include "frc971/control_loops/control_loops_generated.h"
#include "frc971/control_loops/drivetrain/drivetrain_config.h"

namespace y2021_bot3::actors {

class AutonomousActor : public ::frc971::autonomous::BaseAutonomousActor {
 public:
  explicit AutonomousActor(::aos::EventLoop *event_loop);

  bool RunAction(
      const ::frc971::autonomous::AutonomousActionParams *params) override;

 private:
  void Reset();
};

}  // namespace y2021_bot3::actors

#endif  // Y2021_BOT3_ACTORS_AUTONOMOUS_ACTOR_H_
