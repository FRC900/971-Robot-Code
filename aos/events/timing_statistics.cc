#include "aos/events/timing_statistics.h"

#include "aos/events/event_loop_generated.h"
#include "glog/logging.h"

namespace aos {
namespace internal {

void RawFetcherTiming::set_timing_report(timing::Fetcher *new_fetcher) {
  fetcher = new_fetcher;
  if (!new_fetcher) {
    latency.set_statistic(nullptr);
  } else {
    latency.set_statistic(fetcher->mutable_latency());
  }
}

void RawFetcherTiming::ResetTimingReport() {
  if (!fetcher) {
    return;
  }

  latency.Reset();
  fetcher->mutate_count(0);
}

void RawSenderTiming::set_timing_report(timing::Sender *new_sender) {
  sender = new_sender;
  if (!sender) {
    size.set_statistic(nullptr);
  } else {
    size.set_statistic(sender->mutable_size());
  }
}

void RawSenderTiming::ResetTimingReport() {
  if (!sender) {
    return;
  }

  size.Reset();
  sender->mutate_count(0);
  for (size_t ii = 0; ii < kNumErrors; ++ii) {
    sender->mutable_error_counts()->GetMutableObject(ii)->mutate_count(0);
  }
}

void RawSenderTiming::IncrementError(timing::SendError error) {
  if (!sender) {
    return;
  }
  const size_t index = static_cast<size_t>(error);
  timing::SendErrorCount *counter =
      sender->mutable_error_counts()->GetMutableObject(index);
  counter->mutate_count(counter->count() + 1);
}

void TimerTiming::set_timing_report(timing::Timer *new_timer) {
  timer = new_timer;
  if (!timer) {
    wakeup_latency.set_statistic(nullptr);
    handler_time.set_statistic(nullptr);
  } else {
    wakeup_latency.set_statistic(timer->mutable_wakeup_latency());
    handler_time.set_statistic(timer->mutable_handler_time());
  }
}

void TimerTiming::ResetTimingReport() {
  if (!timer) {
    return;
  }

  wakeup_latency.Reset();
  handler_time.Reset();
  timer->mutate_count(0);
}

}  // namespace internal
}  // namespace aos
