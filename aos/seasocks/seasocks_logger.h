#ifndef AOS_SEASOCKS_SEASOCKS_LOGGER_H_
#define AOS_SEASOCKS_SEASOCKS_LOGGER_H_

#include "seasocks/PrintfLogger.h"

namespace aos::seasocks {

class SeasocksLogger : public ::seasocks::PrintfLogger {
 public:
  SeasocksLogger(::seasocks::Logger::Level min_level_to_log)
      : PrintfLogger(min_level_to_log) {}
  void log(::seasocks::Logger::Level level, const char *message) override;
};

}  // namespace aos::seasocks

#endif  // AOS_SEASOCKS_SEASOCKS_LOGGER_H_
