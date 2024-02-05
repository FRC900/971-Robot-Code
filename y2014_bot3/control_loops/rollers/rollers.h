#ifndef Y2014_BOT3_CONTROL_LOOPS_ROLLERS_H_
#define Y2014_BOT3_CONTROL_LOOPS_ROLLERS_H_

#include "frc971/control_loops/control_loop.h"
#include "y2014_bot3/control_loops/rollers/rollers_goal_generated.h"
#include "y2014_bot3/control_loops/rollers/rollers_output_generated.h"
#include "y2014_bot3/control_loops/rollers/rollers_position_generated.h"
#include "y2014_bot3/control_loops/rollers/rollers_status_generated.h"

namespace y2014_bot3::control_loops::rollers {

class Rollers
    : public frc971::controls::ControlLoop<Goal, Position, Status, Output> {
 public:
  // Constructs a control loops which can take a rollers or defaults to the
  // rollers at ::2014_bot3::control_loops::rollers.
  explicit Rollers(::aos::EventLoop *event_loop,
                   const ::std::string &name = "/rollers");

 protected:
  // Executes one cycle of the control loop.
  void RunIteration(const Goal *goal, const Position *position,
                    aos::Sender<Output>::Builder *output,
                    aos::Sender<Status>::Builder *status) override;
};

}  // namespace y2014_bot3::control_loops::rollers

#endif  // Y2014_BOT3_CONTROL_LOOPS_ROLLERS_H_
