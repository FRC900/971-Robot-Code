#include "aos/logging/dynamic_logging.h"

#include <ostream>
#include <string_view>

#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "flatbuffers/string.h"

namespace aos::logging {

DynamicLogging::DynamicLogging(aos::EventLoop *event_loop)
    : application_name_(event_loop->name()) {
  if (event_loop->GetChannel<DynamicLogCommand>("/aos") == nullptr) {
    LOG(WARNING) << "Disabling dynamic logger because the DynamicLogCommand "
                    "channel is not configured.";
  } else {
    event_loop->MakeWatcher("/aos", [this](const DynamicLogCommand &cmd) {
      HandleDynamicLogCommand(cmd);
    });
  }
}

void DynamicLogging::HandleDynamicLogCommand(const DynamicLogCommand &command) {
  // For now we expect someone to do an aos_send at the command line, thecommand
  // may be malformed.
  if (!command.has_name() || !command.has_vlog_level()) return;

  if (command.name()->string_view() != application_name_) {
    return;
  }
  if (command.vlog_level() < 0) {
    return;
  }
  absl::SetGlobalVLogLevel(command.vlog_level());
}

}  // namespace aos::logging
