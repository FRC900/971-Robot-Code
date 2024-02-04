#ifndef Y2023_CONTROL_LOOPS_SUPERSTRUCTURE_END_EFFECTOR_H_
#define Y2023_CONTROL_LOOPS_SUPERSTRUCTURE_END_EFFECTOR_H_

#include "aos/events/event_loop.h"
#include "aos/time/time.h"
#include "frc971/control_loops/control_loop.h"
#include "y2023/constants.h"
#include "y2023/control_loops/superstructure/superstructure_goal_generated.h"
#include "y2023/control_loops/superstructure/superstructure_status_generated.h"
#include "y2023/vision/game_pieces_generated.h"

namespace y2023::control_loops::superstructure {

class EndEffector {
 public:
  static constexpr double kRollerConeSuckVoltage() { return 12.0; }
  static constexpr double kRollerConeSpitVoltage() { return -9.0; }

  static constexpr double kRollerCubeSuckVoltage() { return -7.0; }
  static constexpr double kRollerCubeSpitVoltage() { return 3.0; }

  EndEffector();
  void RunIteration(const ::aos::monotonic_clock::time_point timestamp,
                    RollerGoal roller_goal, double falcon_current,
                    double cone_position, bool beambreak,
                    double *intake_roller_voltage, bool preloaded_with_cone);
  EndEffectorState state() const { return state_; }
  vision::Class game_piece() const { return game_piece_; }
  void Reset();

 private:
  EndEffectorState state_;
  vision::Class game_piece_;

  aos::monotonic_clock::time_point timer_;

  bool beambreak_;
};

}  // namespace y2023::control_loops::superstructure

#endif  // Y2023_CONTROL_LOOPS_SUPERSTRUCTURE_END_EFFECTOR_H_
