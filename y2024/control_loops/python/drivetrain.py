#!/usr/bin/python3

from __future__ import print_function
from frc971.control_loops.python import drivetrain
from frc971.control_loops.python import control_loop
import sys

import gflags
import glog

FLAGS = gflags.FLAGS

gflags.DEFINE_bool('plot', False, 'If true, plot the loop response.')

kDrivetrain = drivetrain.DrivetrainParams(
    J=6.5,
    mass=68.0,
    # TODO(austin): Measure radius a bit better.
    robot_radius=0.39,
    wheel_radius=2.5 * 0.0254,
    motor_type=control_loop.KrakenFOC(),
    num_motors=3,
    G=(14.0 / 52.0) * (26.0 / 58.0),
    q_pos=0.24,
    q_vel=2.5,
    efficiency=0.92,
    has_imu=False,
    force=True,
    kf_q_voltage=1.0,
    controller_poles=[0.82, 0.82])


def main(argv):
    argv = FLAGS(argv)
    glog.init()

    if FLAGS.plot:
        drivetrain.PlotDrivetrainMotions(kDrivetrain)
    elif len(argv) != 5:
        print("Expected .h file name and .cc file name")
    else:
        # Write the generated constants out to a file.
        drivetrain.WriteDrivetrain(argv[1:3], argv[3:5], 'y2024', kDrivetrain)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
