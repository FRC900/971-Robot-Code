#ifndef FRC971_CONSTANTS_H_
#define FRC971_CONSTANTS_H_

#include <cstddef>

namespace frc971 {
namespace constants {

struct HallEffectZeroingConstants {
  // The absolute position of the lower edge of the hall effect sensor.
  double lower_hall_position;
  // The absolute position of the upper edge of the hall effect sensor.
  double upper_hall_position;
  // The difference in scaled units between two hall effect edges.  This is the
  // number of units/cycle.
  double index_difference;
  // Number of cycles we need to see the hall effect high.
  size_t hall_trigger_zeroing_length;
  // Direction the system must be moving in order to zero. True is positive,
  // False is negative direction.
  bool zeroing_move_direction;
};

struct PotAndIndexPulseZeroingConstants {
  // The number of samples in the moving average filter.
  size_t average_filter_size;
  // The difference in scaled units between two index pulses.
  double index_difference;
  // The absolute position in scaled units of one of the index pulses.
  double measured_index_position;
  // Value between 0 and .5 which determines a fraction of the index_diff
  // you want to use.
  double allowable_encoder_error;
};

struct EncoderPlusIndexZeroingConstants {
  // The amount of index pulses in the joint's range of motion.
  int index_pulse_count;
  // The difference in scaled units between two index pulses.
  double index_difference;
  // The absolute position in scaled units of one of the index pulses.
  double measured_index_position;
  // The index pulse that is known, going from lowest in the range of motion to
  // highest (Starting at 0).
  int known_index_pulse;
  // Value between 0 and 0.5 which determines a fraction of the index_diff
  // you want to use. If an index pulse deviates by more than this amount from
  // where we expect to see one then we flag an error.
  double allowable_encoder_error;
};

struct PotAndAbsoluteEncoderZeroingConstants {
  // The number of samples in the moving average filter.
  size_t average_filter_size;
  // The distance that the absolute encoder needs to complete a full rotation.
  double one_revolution_distance;
  // Measured absolute position of the encoder when at zero.
  double measured_absolute_position;

  // Treshold for deciding if we are moving. moving_buffer_size samples need to
  // be within this distance of each other before we use the middle one to zero.
  double zeroing_threshold;
  // Buffer size for deciding if we are moving.
  size_t moving_buffer_size;

  // Value between 0 and 1 indicating what fraction of one_revolution_distance
  // it is acceptable for the offset to move.
  double allowable_encoder_error;
};

struct RelativeEncoderZeroingConstants {};

struct ContinuousAbsoluteEncoderZeroingConstants {
  // The number of samples in the moving average filter.
  size_t average_filter_size;
  // The distance that the absolute encoder needs to complete a full rotation.
  // It is presumed that this will always be 2 * pi for any subsystem using this
  // class, unless you have a continuous system that for some reason doesn't
  // have a logical period of 1 revolution in radians.
  double one_revolution_distance;
  // Measured absolute position of the encoder when at zero.
  double measured_absolute_position;

  // Threshold for deciding if we are moving. moving_buffer_size samples need to
  // be within this distance of each other before we use the middle one to zero.
  double zeroing_threshold;
  // Buffer size for deciding if we are moving.
  size_t moving_buffer_size;

  // Value between 0 and 1 indicating what fraction of a revolution
  // it is acceptable for the offset to move.
  double allowable_encoder_error;
};

struct AbsoluteEncoderZeroingConstants {
  // The number of samples in the moving average filter.
  size_t average_filter_size;
  // The distance that the absolute encoder needs to complete a full rotation.
  double one_revolution_distance;
  // Measured absolute position of the encoder when at zero.
  double measured_absolute_position;
  // Position of the middle of the range of motion in output coordinates.
  double middle_position;

  // Threshold for deciding if we are moving. moving_buffer_size samples need to
  // be within this distance of each other before we use the middle one to zero.
  double zeroing_threshold;
  // Buffer size for deciding if we are moving.
  size_t moving_buffer_size;

  // Value between 0 and 1 indicating what fraction of one_revolution_distance
  // it is acceptable for the offset to move.
  double allowable_encoder_error;
};

struct AbsoluteAndAbsoluteEncoderZeroingConstants {
  // The number of samples in the moving average filter.
  size_t average_filter_size;
  // The distance that the absolute encoder needs to complete a full rotation.
  double one_revolution_distance;
  // Measured absolute position of the encoder when at zero.
  double measured_absolute_position;

  // The distance that the single turn absolute encoder needs to complete a full
  // rotation.
  double single_turn_one_revolution_distance;
  // Measured absolute position of the single turn encoder when at zero.
  double single_turn_measured_absolute_position;
  // Position of the middle of the range of motion in output coordinates.
  double single_turn_middle_position;

  // Threshold for deciding if we are moving. moving_buffer_size samples need to
  // be within this distance of each other before we use the middle one to zero.
  double zeroing_threshold;
  // Buffer size for deciding if we are moving.
  size_t moving_buffer_size;

  // Value between 0 and 1 indicating what fraction of one_revolution_distance
  // it is acceptable for the offset to move.
  double allowable_encoder_error;
};

// Defines a range of motion for a subsystem.
// These are all absolute positions in scaled units.
struct Range {
  double lower_hard;
  double upper_hard;
  double lower;
  double upper;

  constexpr double middle() const { return (lower_hard + upper_hard) / 2.0; }
  constexpr double middle_soft() const { return (lower + upper) / 2.0; }

  constexpr double range() const { return upper_hard - lower_hard; }
};

}  // namespace constants
}  // namespace frc971

#endif  // FRC971_CONSTANTS_H_
