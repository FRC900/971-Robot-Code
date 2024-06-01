#ifndef AOS_LOGGING_IMPLEMENTATIONS_H_
#define AOS_LOGGING_IMPLEMENTATIONS_H_

#include <sys/types.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "aos/logging/context.h"
#include "aos/logging/interface.h"
#include "aos/logging/logging.h"
#include "aos/macros.h"
#include "aos/time/time.h"

// This file has various concrete LogImplementations.

namespace aos {
namespace logging {

// Unless explicitly stated otherwise, format must always be a string constant,
// args are printf-style arguments for format, and ap is a va_list of args.
// The validity of format and args together will be checked at compile time
// using a function attribute.

// Contains all of the information about a given logging call.
struct LogMessage {
  int32_t seconds, nseconds;
  // message_length is just the length of the actual data (which member depends
  // on the type).
  size_t message_length, name_length;
  pid_t source;
  static_assert(sizeof(source) == 4, "that's how they get printed");
  // Per task/thread.
  uint16_t sequence;
  log_level level;
  char name[LOG_MESSAGE_NAME_LEN];
  char message[LOG_MESSAGE_LEN];
};

// Returns left > right. LOG_UNKNOWN is most important.
static inline bool log_gt_important(log_level left, log_level right) {
  if (left == ERROR) left = 3;
  if (right == ERROR) right = 3;
  return left > right;
}

// Returns a string representing level or "unknown".
static inline const char *log_str(log_level level) {
#define DECL_LEVEL(name, value) \
  if (level == name) return #name;
  DECL_LEVELS;
#undef DECL_LEVEL
  return "unknown";
}
// Returns the log level represented by str or LOG_UNKNOWN.
static inline log_level str_log(const char *str) {
#define DECL_LEVEL(name, value) \
  if (!strcmp(str, #name)) return name;
  DECL_LEVELS;
#undef DECL_LEVEL
  return LOG_UNKNOWN;
}

// Implements all of the DoLog* methods in terms of a (pure virtual in this
// class) HandleMessage method that takes a pointer to the message.
class HandleMessageLogImplementation : public LogImplementation {
 protected:
  virtual ::aos::monotonic_clock::time_point monotonic_now() const {
    return ::aos::monotonic_clock::now();
  }

 private:
  __attribute__((format(GOOD_PRINTF_FORMAT_TYPE, 3, 0))) void DoLog(
      log_level level, const char *format, va_list ap) override;

  virtual void HandleMessage(const LogMessage &message) = 0;
};

// A log implementation that dumps all messages to a C stdio stream.
class StreamLogImplementation : public HandleMessageLogImplementation {
 public:
  StreamLogImplementation(FILE *stream);

  // Returns the name of this actual thread as the name.
  std::string_view MyName() override {
    internal::Context *context = internal::Context::Get();
    return context->MyName();
  }

 private:
  void HandleMessage(const LogMessage &message) override;

  FILE *const stream_;
};

// Returns the current implementation.
std::shared_ptr<LogImplementation> GetImplementation();

// Sets the current implementation.
void SetImplementation(std::shared_ptr<LogImplementation> implementation);

// A logging implementation which just uses a callback.
class CallbackLogImplementation : public HandleMessageLogImplementation {
 public:
  CallbackLogImplementation(
      const ::std::function<void(const LogMessage &)> &callback,
      const std::string *name)
      : callback_(callback), name_(name) {}

  // Returns the provided name.  This is most likely the event loop name.
  std::string_view MyName() override { return *name_; }

 private:
  void HandleMessage(const LogMessage &message) override { callback_(message); }

  ::std::function<void(const LogMessage &)> callback_;
  const std::string *name_;
};

class ScopedLogRestorer {
 public:
  ScopedLogRestorer() : prev_impl_(GetImplementation()) {}
  ~ScopedLogRestorer() { SetImplementation(std::move(prev_impl_)); }

  void Swap(std::shared_ptr<LogImplementation> new_impl) {
    SetImplementation(std::move(new_impl));
  }

 private:
  std::shared_ptr<LogImplementation> prev_impl_;
};

// This is where all of the code that is only used by actual LogImplementations
// goes.
namespace internal {

// Fills in *message according to the given inputs (with type kString).
// Used for implementing LogImplementation::DoLog.
void FillInMessage(log_level level, std::string_view name,
                   ::aos::monotonic_clock::time_point monotonic_now,
                   const char *format, va_list ap, LogMessage *message)
    __attribute__((format(GOOD_PRINTF_FORMAT_TYPE, 4, 0)));

__attribute__((format(GOOD_PRINTF_FORMAT_TYPE, 5, 6))) static inline void
FillInMessageVarargs(log_level level, std::string_view name,
                     ::aos::monotonic_clock::time_point monotonic_now,
                     LogMessage *message, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  FillInMessage(level, name, monotonic_now, format, ap, message);
  va_end(ap);
}

// Prints message to output.
void PrintMessage(FILE *output, const LogMessage &message);

}  // namespace internal
}  // namespace logging
}  // namespace aos

#endif  // AOS_LOGGING_IMPLEMENTATIONS_H_
