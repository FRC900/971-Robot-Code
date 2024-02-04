#include "y2024/vision/vision_util.h"

#include "glog/logging.h"

namespace y2024::vision {

const frc971::vision::calibration::CameraCalibration *FindCameraCalibration(
    const y2024::Constants &calibration_data, std::string_view node_name) {
  CHECK(calibration_data.has_cameras());
  for (const y2024::CameraConfiguration *candidate :
       *calibration_data.cameras()) {
    CHECK(candidate->has_calibration());
    if (candidate->calibration()->node_name()->string_view() != node_name) {
      continue;
    }
    return candidate->calibration();
  }
  LOG(FATAL) << ": Failed to find camera calibration for " << node_name;
}

}  // namespace y2024::vision
