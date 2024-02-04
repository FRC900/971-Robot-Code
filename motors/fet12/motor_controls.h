#ifndef MOTORS_MOTOR_CONTROLS_H_
#define MOTORS_MOTOR_CONTROLS_H_

#include <array>
#include <complex>

#include "Eigen/Dense"

#include "motors/math.h"
#include "motors/motor.h"

namespace frc971::motors {

class MotorControlsImplementation : public MotorControls {
 public:
  template <int kRows, int kCols>
  using ComplexMatrix = ::Eigen::Matrix<::std::complex<float>, kRows, kCols>;

  MotorControlsImplementation();
  ~MotorControlsImplementation() override = default;

  void Reset() override {
    estimated_velocity_ = 0;
    filtered_current_ = 0;
  }

  static constexpr int constant_counts_per_revolution() { return 2048; }

  int mechanical_counts_per_revolution() const override {
    return constant_counts_per_revolution();
  }
  int electrical_counts_per_revolution() const override {
    return constant_counts_per_revolution();
  }
  float scale_current_reading(float reading) const override { return reading; }

  ::std::array<float, 3> DoIteration(const float raw_currents[3],
                                     uint32_t theta,
                                     const float command_current) override;

  int16_t Debug(uint32_t theta) override;

  float estimated_velocity() const override { return estimated_velocity_; }

  int16_t i_goal(size_t ii) const override {
    return static_cast<int16_t>(I_last_[ii] * 10.0f);
  }

  float overall_measured_current() const { return overall_measured_current_; }

 private:
  const ComplexMatrix<3, 1> E1Unrotated_, E2Unrotated_;

  float estimated_velocity_ = 0;
  float filtered_current_ = 0;
  float overall_measured_current_ = 0;

  ::Eigen::Matrix<float, 3, 1> I_last_ = ::Eigen::Matrix<float, 3, 1>::Zero();
  ::Eigen::Matrix<float, 3, 1> I_prev_ = ::Eigen::Matrix<float, 3, 1>::Zero();

  int16_t debug_[9];
};

}  // namespace frc971::motors

#endif  // MOTORS_MOTOR_CONTROLS_H_
