#ifndef AOS_SCOPED_SCOPED_FD_H_
#define AOS_SCOPED_SCOPED_FD_H_

#include <unistd.h>

#include "aos/macros.h"

namespace aos {

// Smart "pointer" (container) for a file descriptor.
class ScopedFD {
 public:
  explicit ScopedFD(int fd = -1) : fd_(fd) {}
  ScopedFD(ScopedFD &) = delete;
  ScopedFD(ScopedFD &&other) : ScopedFD(other.release()) {}

  void operator=(const ScopedFD &) = delete;
  void operator=(ScopedFD &&other) {
    int tmp = fd_;
    fd_ = other.fd_;
    other.fd_ = tmp;
  }

  ~ScopedFD() { Close(); }

  int get() const { return fd_; }
  int release() {
    const int r = fd_;
    fd_ = -1;
    return r;
  }
  void reset(int new_fd = -1) {
    if (fd_ != new_fd) {
      Close();
      fd_ = new_fd;
    }
  }
  explicit operator bool() const { return fd_ != -1; }

 private:
  int fd_;

  void Close();
};

}  // namespace aos

#endif  // AOS_SCOPED_SCOPED_FD_H_
