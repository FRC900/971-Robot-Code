#ifndef AOS_EVENTS_PONG_LIB_H_
#define AOS_EVENTS_PONG_LIB_H_

#include "aos/events/event_loop.h"
#include "aos/events/ping_static.h"
#include "aos/events/pong_static.h"

namespace aos {

// Class which replies to a Ping message with a Pong message immediately.
class Pong {
 public:
  Pong(EventLoop *event_loop);

  void set_quiet(bool quiet) { quiet_ = quiet; }

 private:
  void HandlePing(const examples::Ping &ping);
  EventLoop *event_loop_;
  aos::Fetcher<examples::Ping> fetcher_;
  aos::Sender<examples::PongStatic> sender_;
  int32_t last_value_ = 0;
  int32_t last_send_time_ = 0;

  bool quiet_ = true;
};

}  // namespace aos

#endif  // AOS_EVENTS_PONG_LIB_H_
