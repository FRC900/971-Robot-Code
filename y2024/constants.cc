#include "y2024/constants.h"

#include <cinttypes>
#include <map>

#if __has_feature(address_sanitizer)
#include "sanitizer/lsan_interface.h"
#endif

#include "absl/base/call_once.h"
#include "glog/logging.h"

#include "aos/mutex/mutex.h"
#include "aos/network/team_number.h"

namespace y2024 {
namespace constants {

Values MakeValues(uint16_t team) {
  LOG(INFO) << "creating a Constants for team: " << team;

  Values r;

  switch (team) {
    // A set of constants for tests.
    case 1:
      break;

    case kCompTeamNumber:
      break;

    case kPracticeTeamNumber:
      break;

    case kCodingRobotTeamNumber:
      break;

    default:
      LOG(FATAL) << "unknown team: " << team;

      // TODO(milind): add pot range checks once we add ranges
  }

  return r;
}

Values MakeValues() { return MakeValues(aos::network::GetTeamNumber()); }

}  // namespace constants
}  // namespace y2024
