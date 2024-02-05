#ifndef Y2023_CONTROL_LOOPS_SUPERSTRUCTURE_ARM_ARM_CONSTANTS_H_
#define Y2023_CONTROL_LOOPS_SUPERSTRUCTURE_ARM_ARM_CONSTANTS_H_

#include "frc971/control_loops/double_jointed_arm/dynamics.h"

namespace y2023::control_loops::superstructure::arm {

constexpr double kEfficiencyTweak = 0.95;
constexpr double kStallTorque = 4.69 * kEfficiencyTweak;
constexpr double kFreeSpeed = (6380.0 / 60.0) * 2.0 * M_PI;
constexpr double kStallCurrent = 257.0;

constexpr ::frc971::control_loops::arm::ArmConstants kArmConstants = {
    .l0 = 20 * 0.0254,
    .l1 = 38 * 0.0254,

    .m0 = 4.5 / 2.2,
    .m1 = 15.5 / 2.2,

    // Moment of inertia of the joints in kg m^2
    .j0 = 0.092,
    .j1 = 1.6,

    // Radius of the center of mass of the joints in meters.
    .r0 = 4.5 * 0.0254,
    .r1 = 20 * 0.0254,

    // Gear ratios for the two joints.
    .g0 = (60. / 16.) * (58. / 24.) * (95. / 15.),
    .g1 = (52. / 14.) * (42. / 34.) * (66. / 18.) * (96. / 15.),

    // Falcon motor constants.
    .efficiency_tweak = kEfficiencyTweak,
    .stall_torque = kStallTorque,
    .free_speed = kFreeSpeed,
    .stall_current = kStallCurrent,
    .resistance = 12.0 / kStallCurrent,
    .Kv = kFreeSpeed / 12.0,
    .Kt = kStallTorque / kStallCurrent,

    // Number of motors on the distal joint.
    .num_distal_motors = 1.0,
};

}  // namespace y2023::control_loops::superstructure::arm

#endif  // Y2023_CONTROL_LOOPS_SUPERSTRUCTURE_ARM_ARM_CONSTANTS_H_
