#ifndef AOS_FTRACE_H_
#define AOS_FTRACE_H_

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <string_view>

namespace aos {

// Manages interacting with ftrace. Silently hides many errors, because they are
// expected to occur when ftrace is not enabled, and we want calling code to
// continue working in that case.
class Ftrace {
 public:
  Ftrace();
  ~Ftrace();

  // Writes a message with a printf-style format.
  //
  // Silently does nothing if tracing is disabled.
  void FormatMessage(const char *format, ...)
      __attribute__((format(__printf__, 2, 3)));

  // Writes a preformatted message.
  //
  // Silently does nothing if tracing is disabled.
  void WriteMessage(std::string_view content);

  // Turns tracing off, or CHECK-fails if tracing is inaccessible. Does nothing
  // if tracing is currently available but off.
  void TurnOffOrDie();

 private:
  int message_fd_, on_fd_;
};

}  // namespace aos

#endif  // AOS_FTRACE_H_
