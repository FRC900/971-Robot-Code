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
  double k4;
  double k5;
  double k6;
  // Are we using 5 or 8 parameter model?
  int num_params;
};
}

#endif

