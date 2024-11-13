#ifndef FRC971_ORIN_CAMERA_MATRIX_H_
#define FRC971_ORIN_CAMERA_MATRIX_H_

namespace frc971::apriltag
{

struct CameraMatrix {
  double fx;
  double cx;
  double fy;
  double cy;
};

struct DistCoeffs {
  double k1;
  double k2;
  double p1;
  double p2;
  double k3;
};
}

#endif

