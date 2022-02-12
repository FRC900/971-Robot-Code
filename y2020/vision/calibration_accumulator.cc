#include "y2020/vision/calibration_accumulator.h"

#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Eigen/Dense"
#include "aos/events/simulated_event_loop.h"
#include "aos/time/time.h"
#include "frc971/control_loops/quaternion_utils.h"
#include "frc971/wpilib/imu_batch_generated.h"
#include "y2020/vision/charuco_lib.h"

DEFINE_bool(display_undistorted, false,
            "If true, display the undistorted image.");

namespace frc971 {
namespace vision {
using aos::distributed_clock;
using aos::monotonic_clock;
namespace chrono = std::chrono;

constexpr double kG = 9.807;

void CalibrationData::AddCameraPose(
    distributed_clock::time_point distributed_now, Eigen::Vector3d rvec,
    Eigen::Vector3d tvec) {
  // Always start with IMU reading...
  if (!imu_points_.empty() && imu_points_[0].first < distributed_now) {
    rot_trans_points_.emplace_back(distributed_now, std::make_pair(rvec, tvec));
  }
}

void CalibrationData::AddImu(distributed_clock::time_point distributed_now,
                             Eigen::Vector3d gyro, Eigen::Vector3d accel) {
  imu_points_.emplace_back(distributed_now, std::make_pair(gyro, accel));
}

void CalibrationData::ReviewData(CalibrationDataObserver *observer) {
  size_t next_imu_point = 0;
  size_t next_camera_point = 0;
  while (true) {
    if (next_imu_point != imu_points_.size()) {
      // There aren't that many combinations, so just brute force them all
      // rather than being too clever.
      if (next_camera_point != rot_trans_points_.size()) {
        if (imu_points_[next_imu_point].first >
            rot_trans_points_[next_camera_point].first) {
          // Camera!
          observer->UpdateCamera(rot_trans_points_[next_camera_point].first,
                                 rot_trans_points_[next_camera_point].second);
          ++next_camera_point;
        } else {
          // IMU!
          observer->UpdateIMU(imu_points_[next_imu_point].first,
                              imu_points_[next_imu_point].second);
          ++next_imu_point;
        }
      } else {
        if (next_camera_point != rot_trans_points_.size()) {
          // Camera!
          observer->UpdateCamera(rot_trans_points_[next_camera_point].first,
                                 rot_trans_points_[next_camera_point].second);
          ++next_camera_point;
        } else {
          // Nothing left for either list of points, so we are done.
          break;
        }
      }
    }
  }
}

Calibration::Calibration(aos::SimulatedEventLoopFactory *event_loop_factory,
                         aos::EventLoop *image_event_loop,
                         aos::EventLoop *imu_event_loop, std::string_view pi,
                         CalibrationData *data)
    : image_event_loop_(image_event_loop),
      image_factory_(event_loop_factory->GetNodeEventLoopFactory(
          image_event_loop_->node())),
      imu_event_loop_(imu_event_loop),
      imu_factory_(
          event_loop_factory->GetNodeEventLoopFactory(imu_event_loop_->node())),
      charuco_extractor_(
          image_event_loop_, pi,
          [this](cv::Mat rgb_image, monotonic_clock::time_point eof,
                 std::vector<int> charuco_ids,
                 std::vector<cv::Point2f> charuco_corners, bool valid,
                 Eigen::Vector3d rvec_eigen, Eigen::Vector3d tvec_eigen) {
            HandleCharuco(rgb_image, eof, charuco_ids, charuco_corners, valid,
                          rvec_eigen, tvec_eigen);
          }),
      data_(data) {
  imu_factory_->OnShutdown([]() { cv::destroyAllWindows(); });

  imu_event_loop_->MakeWatcher(
      "/drivetrain", [this](const frc971::IMUValuesBatch &imu) {
        if (!imu.has_readings()) {
          return;
        }
        for (const frc971::IMUValues *value : *imu.readings()) {
          HandleIMU(value);
        }
      });
}

void Calibration::HandleCharuco(cv::Mat rgb_image,
                                const monotonic_clock::time_point eof,
                                std::vector<int> /*charuco_ids*/,
                                std::vector<cv::Point2f> /*charuco_corners*/,
                                bool valid, Eigen::Vector3d rvec_eigen,
                                Eigen::Vector3d tvec_eigen) {
  if (valid) {
    data_->AddCameraPose(image_factory_->ToDistributedClock(eof), rvec_eigen,
                         tvec_eigen);

    // TODO(austin): Need a gravity vector input.
    //
    // TODO(austin): Need a state, covariance, and model.
    //
    // TODO(austin): Need to record all the values out of a log and run it
    // as a batch run so we can feed it into ceres.

    // Z -> up
    // Y -> away from cameras 2 and 3
    // X -> left
    Eigen::Vector3d imu(last_value_.accelerometer_x,
                        last_value_.accelerometer_y,
                        last_value_.accelerometer_z);

    Eigen::Quaternion<double> imu_to_camera(
        Eigen::AngleAxisd(-0.5 * M_PI, Eigen::Vector3d::UnitX()));

    Eigen::Quaternion<double> board_to_world(
        Eigen::AngleAxisd(0.5 * M_PI, Eigen::Vector3d::UnitX()));

    Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ",\n", "[", "]",
                             "[", "]");

    const double age_double =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            image_event_loop_->monotonic_now() - eof)
            .count();
    LOG(INFO) << std::fixed << std::setprecision(6) << "Age: " << age_double
              << ", Pose is R:" << rvec_eigen.transpose().format(HeavyFmt)
              << " T:" << tvec_eigen.transpose().format(HeavyFmt);
  }

  cv::imshow("Display", rgb_image);

  if (FLAGS_display_undistorted) {
    const cv::Size image_size(rgb_image.cols, rgb_image.rows);
    cv::Mat undistorted_rgb_image(image_size, CV_8UC3);
    cv::undistort(rgb_image, undistorted_rgb_image,
                  charuco_extractor_.camera_matrix(),
                  charuco_extractor_.dist_coeffs());

    cv::imshow("Display undist", undistorted_rgb_image);
  }
}

void Calibration::HandleIMU(const frc971::IMUValues *imu) {
  VLOG(1) << "IMU " << imu;
  imu->UnPackTo(&last_value_);
  Eigen::Vector3d gyro(last_value_.gyro_x, last_value_.gyro_y,
                       last_value_.gyro_z);
  Eigen::Vector3d accel(last_value_.accelerometer_x,
                        last_value_.accelerometer_y,
                        last_value_.accelerometer_z);

  data_->AddImu(imu_factory_->ToDistributedClock(monotonic_clock::time_point(
                    chrono::nanoseconds(imu->monotonic_timestamp_ns()))),
                gyro, accel * kG);
}

}  // namespace vision
}  // namespace frc971
