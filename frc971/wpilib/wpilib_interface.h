#ifndef FRC971_WPILIB_WPILIB_INTERFACE_H_
#define FRC971_WPILIB_WPILIB_INTERFACE_H_

#include <cstdint>

#include "aos/events/event_loop.h"
#include "frc971/input/robot_state_generated.h"

namespace frc971::wpilib {

// Sends out a message on ::aos::robot_state.
flatbuffers::Offset<aos::RobotState> PopulateRobotState(
    aos::Sender<::aos::RobotState>::Builder *builder, int32_t my_pid);

}  // namespace frc971::wpilib

#endif  // FRC971_WPILIB_WPILIB_INTERFACE_H_
