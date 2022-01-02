#ifndef Y2019_JEVOIS_SPI_H_
#define Y2019_JEVOIS_SPI_H_

#include <array>
#include <cstdint>
#include <optional>

#include "absl/types/span.h"
#include "y2019/jevois/structures.h"

// This file manages serializing and deserializing the various structures for
// transport via SPI.
//
// Our SPI transfers are fixed-size to simplify everything.

namespace frc971 {
namespace jevois {

constexpr size_t spi_transfer_size() {
  // The teensy->RoboRIO side is way bigger, so just calculate that.
  return 3 /* 3 frames */ *
             (1 /* age */ + 3 /* targets */ * 4 /* target size */) +
         2 /* CRC-16 */;
}
static_assert(spi_transfer_size() == 41, "hand math is wrong");
using SpiTransfer = std::array<char, spi_transfer_size()>;

SpiTransfer SpiPackToRoborio(const TeensyToRoborio &message);
std::optional<TeensyToRoborio> SpiUnpackToRoborio(
    absl::Span<const char> transfer);
SpiTransfer SpiPackToTeensy(const RoborioToTeensy &message);
std::optional<RoborioToTeensy> SpiUnpackToTeensy(
    absl::Span<const char> transfer);

}  // namespace jevois
}  // namespace frc971

#endif  // Y2019_JEVOIS_SPI_H_
