#ifndef Y2019_JOYSTICK_ANGLE_H_
#define Y2019_JOYSTICK_ANGLE_H_

#include "frc971/input/driver_station_data.h"

using ::frc971::input::driver_station::Data;
using ::frc971::input::driver_station::JoystickAxis;

namespace y2019::input::joysticks {
bool AngleCloseTo(double angle, double near, double range);

enum class JoystickAngle {
  kDefault,
  kUpperRight,
  kMiddleRight,
  kLowerRight,
  kUpperLeft,
  kMiddleLeft,
  kLowerLeft
};

JoystickAngle GetJoystickPosition(const JoystickAxis &x_axis,
                                  const JoystickAxis &y_axis, const Data &data);
JoystickAngle GetJoystickPosition(float x_axis, float y_axis);

}  // namespace y2019::input::joysticks

#endif  // Y2019_JOYSTICK_ANGLE_H_
