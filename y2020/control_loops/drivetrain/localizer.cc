#include "y2020/control_loops/drivetrain/localizer.h"

#include "y2020/constants.h"

DEFINE_bool(send_empty_debug, false,
            "If true, send LocalizerDebug messages on every tick, even if "
            "they would be empty.");

namespace y2020::control_loops::drivetrain {

namespace {
// Converts a flatbuffer TransformationMatrix to an Eigen matrix. Technically,
// this should be able to do a single memcpy, but the extra verbosity here seems
// appropriate.
Eigen::Matrix<float, 4, 4> FlatbufferToTransformationMatrix(
    const frc971::vision::sift::TransformationMatrix &flatbuffer) {
  CHECK(flatbuffer.data() != nullptr);
  CHECK_EQ(16u, flatbuffer.data()->size());
  Eigen::Matrix<float, 4, 4> result;
  result.setIdentity();
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      result(row, col) = (*flatbuffer.data())[row * 4 + col];
    }
  }
  return result;
}

// Offset to add to the pi's yaw in its extrinsics, to account for issues in the
// calibrated extrinsics.
constexpr double kTurretPiOffset = 0.0;

// Indices of the pis to use.
const std::array<std::string, 5> kPisToUse{"pi1", "pi2", "pi3", "pi4", "pi5"};

float CalculateYaw(const Eigen::Matrix4f &transform) {
  const Eigen::Vector2f yaw_coords =
      (transform * Eigen::Vector4f(1, 0, 0, 0)).topRows<2>();
  return std::atan2(yaw_coords(1), yaw_coords(0));
}

// Calculates the pose implied by the camera target, just based on
// distance/heading components.
Eigen::Vector3f CalculateImpliedPose(const float correction_robot_theta,
                                     const Eigen::Matrix4f &H_field_target,
                                     const Localizer::Pose &pose_robot_target) {
  // This code overrides the pose sent directly from the camera code and
  // effectively distills it down to just a distance + heading estimate, on
  // the presumption that these signals will tend to be much lower noise and
  // better-conditioned than other portions of the robot pose.
  // As such, this code assumes that the provided estimate of the robot
  // heading is correct and then, given the heading from the camera to the
  // target and the distance from the camera to the target, calculates the
  // position that the robot would have to be at to make the current camera
  // heading + distance correct. This X/Y implied robot position is then
  // used as the measurement in the EKF, rather than the X/Y that is
  // directly returned from the vision processing. If the provided
  // correction_robot_theta is exactly identical to the current estimated robot
  // yaw, then this means that the image corrections will not do anything to
  // correct gyro drift; however, by making that tradeoff we can prioritize
  // getting the turret angle to the target correct (without adding any new
  // non-linearities to the EKF itself).

  // Calculate the heading to the robot in the target's coordinate frame.
  // Reminder on what the numbers mean:
  // rel_theta: The orientation of the target in the robot frame.
  // heading: heading from the robot to the target in the robot frame. I.e.,
  //   atan2(y, x) for x/y of the target in the robot frame.
  const float implied_rel_theta =
      CalculateYaw(H_field_target) - correction_robot_theta;
  const float implied_heading_from_target = aos::math::NormalizeAngle(
      M_PI - implied_rel_theta + pose_robot_target.heading());
  const float implied_distance = pose_robot_target.xy_norm();
  const Eigen::Vector4f robot_pose_in_target_frame(
      implied_distance * std::cos(implied_heading_from_target),
      implied_distance * std::sin(implied_heading_from_target), 0, 1);
  const Eigen::Vector2f implied_xy =
      (H_field_target * robot_pose_in_target_frame).topRows<2>();
  return {implied_xy.x(), implied_xy.y(), correction_robot_theta};
}

}  // namespace

Localizer::Localizer(
    aos::EventLoop *event_loop,
    const frc971::control_loops::drivetrain::DrivetrainConfig<double>
        &dt_config)
    : event_loop_(event_loop),
      dt_config_(dt_config),
      ekf_(dt_config),
      observations_(&ekf_),
      clock_offset_fetcher_(
          event_loop_->MakeFetcher<aos::message_bridge::ServerStatistics>(
              "/aos")),
      debug_sender_(
          event_loop_
              ->MakeSender<y2020::control_loops::drivetrain::LocalizerDebug>(
                  "/drivetrain")) {
  statistics_.rejection_counts.fill(0);
  // TODO(james): The down estimator has trouble handling situations where the
  // robot is constantly wiggling but not actually moving much, and can cause
  // drift when using accelerometer readings.
  ekf_.set_ignore_accel(true);
  // TODO(james): This doesn't really need to be a watcher; we could just use a
  // fetcher for the superstructure status.
  // This probably should be a Fetcher instead of a Watcher, but this
  // seems simpler for the time being (although technically it should be
  // possible to do everything we need to using just a Fetcher without
  // even maintaining a separate buffer, but that seems overly cute).
  event_loop_->MakeWatcher("/superstructure",
                           [this](const superstructure::Status &status) {
                             HandleSuperstructureStatus(status);
                           });

  event_loop->OnRun([this, event_loop]() {
    ekf_.ResetInitialState(event_loop->monotonic_now(),
                           HybridEkf::State::Zero(), ekf_.P());
  });

  for (const auto &pi : kPisToUse) {
    image_fetchers_.emplace_back(
        event_loop_->MakeFetcher<frc971::vision::sift::ImageMatchResult>(
            "/" + pi + "/camera"));
  }

  target_selector_.set_has_target(false);
}

void Localizer::Reset(
    aos::monotonic_clock::time_point t,
    const frc971::control_loops::drivetrain::HybridEkf<double>::State &state) {
  // Go through and clear out all of the fetchers so that we don't get behind.
  for (auto &fetcher : image_fetchers_) {
    fetcher.Fetch();
  }
  ekf_.ResetInitialState(t, state.cast<float>(), ekf_.P());
}

void Localizer::HandleSuperstructureStatus(
    const y2020::control_loops::superstructure::Status &status) {
  CHECK(status.has_turret());
  turret_data_.Push({event_loop_->monotonic_now(), status.turret()->position(),
                     status.turret()->velocity()});
}

Localizer::TurretData Localizer::GetTurretDataForTime(
    aos::monotonic_clock::time_point time) {
  if (turret_data_.empty()) {
    return {};
  }

  aos::monotonic_clock::duration lowest_time_error =
      aos::monotonic_clock::duration::max();
  TurretData best_data_match;
  for (const auto &sample : turret_data_) {
    const aos::monotonic_clock::duration time_error =
        std::chrono::abs(sample.receive_time - time);
    if (time_error < lowest_time_error) {
      lowest_time_error = time_error;
      best_data_match = sample;
    }
  }
  return best_data_match;
}

void Localizer::Update(const Eigen::Matrix<double, 2, 1> &U,
                       aos::monotonic_clock::time_point now,
                       double left_encoder, double right_encoder,
                       double gyro_rate, const Eigen::Vector3d &accel) {
  ekf_.UpdateEncodersAndGyro(left_encoder, right_encoder, gyro_rate,
                             U.cast<float>(), accel.cast<float>(), now);
  auto builder = debug_sender_.MakeBuilder();
  aos::SizedArray<flatbuffers::Offset<ImageMatchDebug>, 25> debug_offsets;
  for (size_t ii = 0; ii < kPisToUse.size(); ++ii) {
    auto &image_fetcher = image_fetchers_[ii];
    while (image_fetcher.FetchNext()) {
      const auto offsets = HandleImageMatch(ii, kPisToUse[ii], *image_fetcher,
                                            now, builder.fbb());
      for (const auto offset : offsets) {
        debug_offsets.push_back(offset);
      }
    }
  }
  if (FLAGS_send_empty_debug || !debug_offsets.empty()) {
    const auto vector_offset =
        builder.fbb()->CreateVector(debug_offsets.data(), debug_offsets.size());
    const auto rejections_offset =
        builder.fbb()->CreateVector(statistics_.rejection_counts.data(),
                                    statistics_.rejection_counts.size());

    CumulativeStatistics::Builder stats_builder =
        builder.MakeBuilder<CumulativeStatistics>();
    stats_builder.add_total_accepted(statistics_.total_accepted);
    stats_builder.add_total_candidates(statistics_.total_candidates);
    stats_builder.add_rejection_reason_count(rejections_offset);
    const auto stats_offset = stats_builder.Finish();

    LocalizerDebug::Builder debug_builder =
        builder.MakeBuilder<LocalizerDebug>();
    debug_builder.add_matches(vector_offset);
    debug_builder.add_statistics(stats_offset);
    builder.CheckOk(builder.Send(debug_builder.Finish()));
  }
}

aos::SizedArray<flatbuffers::Offset<ImageMatchDebug>, 5>
Localizer::HandleImageMatch(
    size_t camera_index, std::string_view pi,
    const frc971::vision::sift::ImageMatchResult &result,
    aos::monotonic_clock::time_point now, flatbuffers::FlatBufferBuilder *fbb) {
  aos::SizedArray<flatbuffers::Offset<ImageMatchDebug>, 5> debug_offsets;

  std::chrono::nanoseconds monotonic_offset{0};
  bool message_bridge_connected = true;
  clock_offset_fetcher_.Fetch();
  if (clock_offset_fetcher_.get() != nullptr) {
    for (const auto connection : *clock_offset_fetcher_->connections()) {
      if (connection->has_node() && connection->node()->has_name() &&
          connection->node()->name()->string_view() == pi) {
        if (connection->has_monotonic_offset()) {
          monotonic_offset =
              std::chrono::nanoseconds(connection->monotonic_offset());
        } else {
          // If we don't have a monotonic offset, that means we aren't
          // connected, in which case we should break the loop but shouldn't
          // populate the offset.
          message_bridge_connected = false;
        }
        break;
      }
    }
  }
  aos::monotonic_clock::time_point capture_time(
      std::chrono::nanoseconds(result.image_monotonic_timestamp_ns()) -
      monotonic_offset);
  VLOG(1) << "Got monotonic offset of "
          << aos::time::DurationInSeconds(monotonic_offset)
          << " when at time of " << now << " and capture time estimate of "
          << capture_time;
  std::optional<RejectionReason> rejection_reason;
  if (!message_bridge_connected) {
    rejection_reason = RejectionReason::MESSAGE_BRIDGE_DISCONNECTED;
  } else if (capture_time > now) {
    rejection_reason = RejectionReason::IMAGE_FROM_FUTURE;
  }
  if (!result.has_camera_calibration()) {
    AOS_LOG(WARNING, "Got camera frame without calibration data.\n");
    ImageMatchDebug::Builder builder(*fbb);
    builder.add_camera(camera_index);
    builder.add_accepted(false);
    builder.add_rejection_reason(RejectionReason::NO_CALIBRATION);
    debug_offsets.push_back(builder.Finish());
    statistics_.rejection_counts[static_cast<size_t>(
        RejectionReason::NO_CALIBRATION)]++;
    return debug_offsets;
  }
  // Per the ImageMatchResult specification, we can actually determine whether
  // the camera is the turret camera just from the presence of the
  // turret_extrinsics member.
  const bool is_turret = result.camera_calibration()->has_turret_extrinsics();
  const TurretData turret_data = GetTurretDataForTime(capture_time);
  // Ignore readings when the turret is spinning too fast, on the assumption
  // that the odds of screwing up the time compensation are higher.
  // Note that the current number here is chosen pretty arbitrarily--1 rad / sec
  // seems reasonable, but may be unnecessarily low or high.
  constexpr float kMaxTurretVelocity = 1.0;
  if (is_turret && std::abs(turret_data.velocity) > kMaxTurretVelocity &&
      !rejection_reason) {
    rejection_reason = RejectionReason::TURRET_TOO_FAST;
  }
  CHECK(result.camera_calibration()->has_fixed_extrinsics());
  const Eigen::Matrix<float, 4, 4> fixed_extrinsics =
      FlatbufferToTransformationMatrix(
          *result.camera_calibration()->fixed_extrinsics());

  // Calculate the pose of the camera relative to the robot origin.
  Eigen::Matrix<float, 4, 4> H_robot_camera = fixed_extrinsics;
  if (is_turret) {
    H_robot_camera = H_robot_camera *
                     frc971::control_loops::TransformationMatrixForYaw<float>(
                         turret_data.position + kTurretPiOffset) *
                     FlatbufferToTransformationMatrix(
                         *result.camera_calibration()->turret_extrinsics());
  }

  if (!result.has_camera_poses()) {
    ImageMatchDebug::Builder builder(*fbb);
    builder.add_camera(camera_index);
    builder.add_accepted(false);
    builder.add_rejection_reason(RejectionReason::NO_RESULTS);
    debug_offsets.push_back(builder.Finish());
    statistics_
        .rejection_counts[static_cast<size_t>(RejectionReason::NO_RESULTS)]++;
    return debug_offsets;
  }

  int index = -1;
  for (const frc971::vision::sift::CameraPose *vision_result :
       *result.camera_poses()) {
    ++statistics_.total_candidates;
    ++index;

    ImageMatchDebug::Builder builder(*fbb);
    builder.add_camera(camera_index);
    builder.add_pose_index(index);
    builder.add_local_image_capture_time_ns(
        result.image_monotonic_timestamp_ns());
    builder.add_roborio_image_capture_time_ns(
        capture_time.time_since_epoch().count());
    builder.add_image_age_sec(aos::time::DurationInSeconds(now - capture_time));

    if (!vision_result->has_camera_to_target() ||
        !vision_result->has_field_to_target()) {
      builder.add_accepted(false);
      builder.add_rejection_reason(RejectionReason::NO_TRANSFORMS);
      statistics_.rejection_counts[static_cast<size_t>(
          RejectionReason::NO_TRANSFORMS)]++;
      debug_offsets.push_back(builder.Finish());
      continue;
    }
    const Eigen::Matrix<float, 4, 4> H_camera_target =
        FlatbufferToTransformationMatrix(*vision_result->camera_to_target());

    const Eigen::Matrix<float, 4, 4> H_field_target =
        FlatbufferToTransformationMatrix(*vision_result->field_to_target());
    const Eigen::Matrix<float, 4, 4> H_field_camera =
        H_field_target * H_camera_target.inverse();
    // Back out the robot position that is implied by the current camera
    // reading. Note that the Pose object ignores any roll/pitch components, so
    // if the camera's extrinsics for pitch/roll are off, this should just
    // ignore it.
    const Pose measured_camera_pose(H_field_camera);
    builder.add_camera_x(measured_camera_pose.rel_pos().x());
    builder.add_camera_y(measured_camera_pose.rel_pos().y());
    // Because the camera uses Z as forwards rather than X, just calculate the
    // debugging theta value using the transformation matrix directly (note that
    // the rest of this file deliberately does not care what convention the
    // camera uses, since that is encoded in the extrinsics themselves).
    builder.add_camera_theta(
        std::atan2(H_field_camera(1, 2), H_field_camera(0, 2)));
    // Calculate the camera-to-robot transformation matrix ignoring the
    // pitch/roll of the camera.
    // TODO(james): This could probably be made a bit more efficient, but I
    // don't think this is anywhere near our bottleneck currently.
    const Eigen::Matrix<float, 4, 4> H_camera_robot_stripped =
        Pose(H_robot_camera).AsTransformationMatrix().inverse();
    const Pose measured_pose(measured_camera_pose.AsTransformationMatrix() *
                             H_camera_robot_stripped);
    // This "Z" is the robot pose directly implied by the camera results.
    // Currently, we do not actually use this result directly. However, it is
    // kept around in case we want to quickly re-enable it.
    const Eigen::Matrix<float, 3, 1> Z(measured_pose.rel_pos().x(),
                                       measured_pose.rel_pos().y(),
                                       measured_pose.rel_theta());
    builder.add_implied_robot_x(Z(0));
    builder.add_implied_robot_y(Z(1));
    builder.add_implied_robot_theta(Z(2));
    // Pose of the target in the robot frame.
    // Note that we use measured_pose's transformation matrix rather than just
    // doing H_robot_camera * H_camera_target because measured_pose ignores
    // pitch/roll.
    Pose pose_robot_target(measured_pose.AsTransformationMatrix().inverse() *
                           H_field_target);

    // Turret is zero when pointed backwards.
    builder.add_implied_turret_goal(
        aos::math::NormalizeAngle(M_PI + pose_robot_target.heading()));

    // Since we've now built up all the information that is useful to include in
    // the debug message, bail if we have reason to do so.
    if (rejection_reason) {
      builder.add_rejection_reason(*rejection_reason);
      statistics_.rejection_counts[static_cast<size_t>(*rejection_reason)]++;
      builder.add_accepted(false);
      debug_offsets.push_back(builder.Finish());
      continue;
    }

    // TODO(james): Figure out how to properly handle calculating the
    // noise. Currently, the values are deliberately tuned so that image updates
    // will not be trusted overly much. In theory, we should probably also be
    // populating some cross-correlation terms.
    // Note that these are the noise standard deviations (they are squared below
    // to get variances).
    Eigen::Matrix<float, 3, 1> noises(2.0, 2.0, 0.5);
    // Augment the noise by the approximate rotational speed of the
    // camera. This should help account for the fact that, while we are
    // spinning, slight timing errors in the camera/turret data will tend to
    // have mutch more drastic effects on the results.
    noises *= 1.0 + std::abs((right_velocity() - left_velocity()) /
                                 (2.0 * dt_config_.robot_radius) +
                             (is_turret ? turret_data.velocity : 0.0));

    // Pay less attention to cameras that aren't actually on the turret, since
    // they are less useful when it comes to actually making shots.
    if (!is_turret) {
      noises *= 3.0;
    } else {
      noises /= 5.0;
    }

    Eigen::Matrix3f R = Eigen::Matrix3f::Zero();
    R.diagonal() = noises.cwiseAbs2();
    VLOG(1) << "Pose implied by target: " << Z.transpose()
            << " and current pose " << x() << ", " << y() << ", " << theta()
            << " Heading/dist/skew implied by target: "
            << pose_robot_target.ToHeadingDistanceSkew().transpose();
    // If the heading is off by too much, assume that we got a false-positive
    // and don't use the correction.
    if (std::abs(aos::math::DiffAngle<float>(theta(), Z(2))) > M_PI_2) {
      AOS_LOG(WARNING, "Dropped image match due to heading mismatch.\n");
      builder.add_accepted(false);
      builder.add_rejection_reason(RejectionReason::HIGH_THETA_DIFFERENCE);
      statistics_.rejection_counts[static_cast<size_t>(
          RejectionReason::HIGH_THETA_DIFFERENCE)]++;
      debug_offsets.push_back(builder.Finish());
      continue;
    }
    // In order to do the EKF correction, we determine the expected state based
    // on the state at the time the image was captured; however, we insert the
    // correction update itself at the current time. This is technically not
    // quite correct, but saves substantial CPU usage by making it so that we
    // don't have to constantly rewind the entire EKF history.
    const std::optional<State> state_at_capture =
        ekf_.LastStateBeforeTime(capture_time);
    if (!state_at_capture.has_value()) {
      AOS_LOG(WARNING, "Dropped image match due to age of image.\n");
      builder.add_accepted(false);
      builder.add_rejection_reason(RejectionReason::IMAGE_TOO_OLD);
      statistics_.rejection_counts[static_cast<size_t>(
          RejectionReason::IMAGE_TOO_OLD)]++;
      debug_offsets.push_back(builder.Finish());
      continue;
    }

    std::optional<RejectionReason> correction_rejection;
    const Input U = ekf_.MostRecentInput();
    // For the correction step, instead of passing in the measurement directly,
    // we pass in (0, 0, 0) as the measurement and then for the expected
    // measurement (Zhat) we calculate the error between the implied and actual
    // poses. This doesn't affect any of the math, it just makes the code a bit
    // more convenient to write given the Correct() interface we already have.
    // Note: If we start going back to doing back-in-time rewinds, then we can't
    // get away with passing things by reference.
    observations_.CorrectKnownH(
        Eigen::Vector3f::Zero(), &U,
        Corrector(H_field_target, pose_robot_target, state_at_capture.value(),
                  Z, &correction_rejection),
        R, now);
    if (correction_rejection) {
      builder.add_accepted(false);
      builder.add_rejection_reason(*correction_rejection);
      statistics_
          .rejection_counts[static_cast<size_t>(*correction_rejection)]++;
    } else {
      builder.add_accepted(true);
      statistics_.total_accepted++;
    }
    debug_offsets.push_back(builder.Finish());
  }
  return debug_offsets;
}

Localizer::Output Localizer::Corrector::H(const State &, const Input &) {
  // Weighting for how much to use the current robot heading estimate
  // vs. the heading estimate from the image results. A value of 1.0
  // completely ignores the measured heading, but will prioritize turret
  // aiming above all else. A value of 0.0 will prioritize correcting
  // any gyro heading drift.
  constexpr float kImpliedWeight = 0.99;
  const float z_yaw_diff = aos::math::NormalizeAngle(
      state_at_capture_(Localizer::StateIdx::kTheta) - Z_(2));
  const float z_yaw = Z_(2) + kImpliedWeight * z_yaw_diff;
  const Eigen::Vector3f Z_implied =
      CalculateImpliedPose(z_yaw, H_field_target_, pose_robot_target_);
  const Eigen::Vector3f Z_used = Z_implied;
  // Just in case we ever do encounter any, drop measurements if they
  // have non-finite numbers.
  if (!Z_.allFinite()) {
    AOS_LOG(WARNING, "Got measurement with infinites or NaNs.\n");
    *correction_rejection_ = RejectionReason::NONFINITE_MEASUREMENT;
    return Eigen::Vector3f::Zero();
  }
  Eigen::Vector3f Zhat = H_ * state_at_capture_ - Z_used;
  // Rewrap angle difference to put it back in range. Note that this
  // component of the error is currently ignored (see definition of H
  // above).
  Zhat(2) = aos::math::NormalizeAngle(Zhat(2));
  // If the measurement implies that we are too far from the current
  // estimate, then ignore it.
  // Note that I am not entirely sure how much effect this actually has,
  // because I primarily introduced it to make sure that any grossly
  // invalid measurements get thrown out.
  if (Zhat.squaredNorm() > std::pow(10.0, 2)) {
    *correction_rejection_ = RejectionReason::CORRECTION_TOO_LARGE;
    return Eigen::Vector3f::Zero();
  }
  return Zhat;
}

}  // namespace y2020::control_loops::drivetrain
