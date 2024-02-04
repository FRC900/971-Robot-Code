#ifndef AOS_UTIL_DEATH_TEST_LOG_IMPLEMENTATION_H_
#define AOS_UTIL_DEATH_TEST_LOG_IMPLEMENTATION_H_

#include <cstdlib>

#include "aos/logging/context.h"
#include "aos/logging/implementations.h"

namespace aos::util {

// Prints all FATAL messages to stderr and then abort(3)s before the regular
// stuff can print out anything else. Ignores all other messages.
// This is useful in death tests that expect a LOG(FATAL) to cause the death.
class DeathTestLogImplementation
    : public logging::HandleMessageLogImplementation {
 public:
  std::string_view MyName() override {
    logging::internal::Context *context = logging::internal::Context::Get();
    return context->MyName();
  }
  virtual void HandleMessage(const logging::LogMessage &message) override {
    if (message.level == FATAL) {
      logging::internal::PrintMessage(stderr, message);
      abort();
    }
  }
};

}  // namespace aos::util

#endif  // AOS_UTIL_DEATH_TEST_LOG_IMPLEMENTATION_H_
