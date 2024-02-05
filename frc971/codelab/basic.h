#ifndef FRC971_CODELAB_BASIC_H_
#define FRC971_CODELAB_BASIC_H_

#include "aos/time/time.h"
#include "frc971/codelab/basic_goal_generated.h"
#include "frc971/codelab/basic_output_generated.h"
#include "frc971/codelab/basic_position_generated.h"
#include "frc971/codelab/basic_status_generated.h"
#include "frc971/control_loops/control_loop.h"

namespace frc971::codelab {

class Basic
    : public ::frc971::controls::ControlLoop<Goal, Position, Status, Output> {
 public:
  explicit Basic(::aos::EventLoop *event_loop,
                 const ::std::string &name = "/codelab");

 protected:
  void RunIteration(const Goal *goal, const Position *position,
                    aos::Sender<Output>::Builder *output,
                    aos::Sender<Status>::Builder *status) override;
};

}  // namespace frc971::codelab

#endif  // FRC971_CODELAB_BASIC_H_
