#ifndef Y2017_CONTROL_LOOPS_SUPERSTRUCTURE_VISION_TIME_ADJUSTER_H_
#define Y2017_CONTROL_LOOPS_SUPERSTRUCTURE_VISION_TIME_ADJUSTER_H_

#include <cstdint>

#include "aos/containers/ring_buffer.h"
#include "aos/events/event_loop.h"
#include "aos/time/time.h"
#include "frc971/control_loops/drivetrain/drivetrain_status_generated.h"
#include "y2017/vision/vision_generated.h"

namespace y2017::control_loops::superstructure {

class VisionTimeAdjuster {
 public:
  VisionTimeAdjuster(::aos::EventLoop *event_loop);

  // This needs to be called at the same interval as the control loops so that
  // it can attempt to make accurate goal recommendations.
  void Tick(::aos::monotonic_clock::time_point monotonic_now,
            double turret_position, const vision::VisionStatus *vision_status);

  // Returns true if we have enough data to recommend a goal for the turret.
  bool valid() const { return valid_; }

  // Returns the goal that we are recommending for the turret. This value is
  // only valid if valid() returns true.
  double goal() const { return goal_; }

  double most_recent_vision_angle() const { return most_recent_vision_angle_; }
  double most_recent_vision_reading() const {
    return most_recent_vision_reading_;
  }

  struct ColumnAngle {
    ::aos::monotonic_clock::time_point time;
    double turret;
  };

  struct DrivetrainAngle {
    ::aos::monotonic_clock::time_point time;
    double left;
    double right;
  };

  void ResetTime() {
    most_recent_vision_time_ = ::aos::monotonic_clock::min_time;
  }

 private:
  // Fetcher to grab the latest drivetrain message.
  ::aos::Fetcher<::frc971::control_loops::drivetrain::Status>
      drivetrain_status_fetcher_;

  // Buffer space to store the most recent drivetrain and turret messages from
  // the last second.
  ::aos::RingBuffer<ColumnAngle, 200> column_data_;
  ::aos::RingBuffer<DrivetrainAngle, 200> drivetrain_data_;

  // The most recently computed goal angle of the turret. This does not yet
  // include the most recent drivetrain angle. Subtract the most recent
  // drivetrain angle from this to get the recommended turret goal.
  double most_recent_vision_angle_ = 0.0;
  ::aos::monotonic_clock::time_point most_recent_vision_time_ =
      ::aos::monotonic_clock::min_time;

  // The most recent angle of the drivetrain.
  double most_recent_drivetrain_angle_ = 0.0;
  double most_recent_vision_reading_ = 0.0;

  double goal_ = 0.0;
  bool valid_ = false;
};

}  // namespace y2017::control_loops::superstructure

#endif  // Y2017_CONTROL_LOOPS_SUPERSTRUCTURE_VISION_TIME_ADJUSTER_H_
