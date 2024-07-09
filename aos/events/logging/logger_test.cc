#include <sys/stat.h>

#include <filesystem>

#include "absl/strings/str_format.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "aos/events/event_loop.h"
#include "aos/events/logging/log_reader.h"
#include "aos/events/logging/log_writer.h"
#include "aos/events/logging/snappy_encoder.h"
#include "aos/events/message_counter.h"
#include "aos/events/ping_lib.h"
#include "aos/events/pong_lib.h"
#include "aos/events/simulated_event_loop.h"
#include "aos/network/remote_message_generated.h"
#include "aos/network/testing_time_converter.h"
#include "aos/network/timestamp_generated.h"
#include "aos/testing/path.h"
#include "aos/testing/tmpdir.h"
#include "aos/util/file.h"

#ifdef LZMA
#include "aos/events/logging/lzma_encoder.h"
#endif

namespace aos::logger::testing {

namespace chrono = std::chrono;
using aos::message_bridge::RemoteMessage;
using aos::testing::ArtifactPath;
using aos::testing::MessageCounter;

constexpr std::string_view kSingleConfigSha256(
    "bbe1b563139273b23a5405eebc2f2740cefcda5f96681acd0a84b8ff9ab93ea4");

class LoggerTest : public ::testing::Test {
 public:
  LoggerTest()
      : config_(aos::configuration::ReadConfig(
            ArtifactPath("aos/events/pingpong_config.json"))),
        event_loop_factory_(&config_.message()),
        ping_event_loop_(event_loop_factory_.MakeEventLoop("ping")),
        ping_(ping_event_loop_.get()),
        pong_event_loop_(event_loop_factory_.MakeEventLoop("pong")),
        pong_(pong_event_loop_.get()) {}

  // Config and factory.
  aos::FlatbufferDetachedBuffer<aos::Configuration> config_;
  SimulatedEventLoopFactory event_loop_factory_;

  // Event loop and app for Ping
  std::unique_ptr<EventLoop> ping_event_loop_;
  Ping ping_;

  // Event loop and app for Pong
  std::unique_ptr<EventLoop> pong_event_loop_;
  Pong pong_;
};

using LoggerDeathTest = LoggerTest;

// Tests that we can startup at all.  This confirms that the channels are all in
// the config.
TEST_F(LoggerTest, Starts) {
  const ::std::string tmpdir = aos::testing::TestTmpDir();
  const ::std::string base_name = tmpdir + "/logfile";
  const ::std::string config =
      absl::StrCat(base_name, kSingleConfigSha256, ".bfbs");
  const ::std::string logfile = base_name + "_data.part0.bfbs";
  // Remove it.
  unlink(config.c_str());
  unlink(logfile.c_str());

  LOG(INFO) << "Logging data to " << logfile;

  {
    event_loop_factory_.RunFor(chrono::milliseconds(95));

    std::unique_ptr<EventLoop> logger_event_loop =
        event_loop_factory_.MakeEventLoop("logger");
    Logger logger(logger_event_loop.get());
    logger.set_separate_config(false);
    logger.set_polling_period(std::chrono::milliseconds(100));
    logger.StartLoggingOnRun(base_name);
    event_loop_factory_.RunFor(chrono::milliseconds(20000));
  }

  ASSERT_TRUE(std::filesystem::exists(logfile));

  // Even though it doesn't make any difference here, exercise the logic for
  // passing in a separate config.
  LogReader reader(logfile, &config_.message());

  // This sends out the fetched messages and advances time to the start of the
  // log file.
  reader.Register();

  EXPECT_THAT(reader.LoggedNodes(), ::testing::ElementsAre(nullptr));

  std::unique_ptr<EventLoop> test_event_loop =
      reader.event_loop_factory()->MakeEventLoop("log_reader");

  int ping_count = 10;
  int pong_count = 10;

  // Confirm that the ping value matches.
  test_event_loop->MakeWatcher("/test",
                               [&ping_count](const examples::Ping &ping) {
                                 EXPECT_EQ(ping.value(), ping_count + 1);
                                 ++ping_count;
                               });
  // Confirm that the ping and pong counts both match, and the value also
  // matches.
  test_event_loop->MakeWatcher(
      "/test", [&pong_count, &ping_count](const examples::Pong &pong) {
        EXPECT_EQ(pong.value(), pong_count + 1);
        ++pong_count;
        EXPECT_EQ(ping_count, pong_count);
      });

  reader.event_loop_factory()->RunFor(std::chrono::seconds(100));
  EXPECT_EQ(ping_count, 2010);
  EXPECT_EQ(pong_count, ping_count);
}

// Tests that we can mutate a message before sending
TEST_F(LoggerTest, MutateCallback) {
  const ::std::string tmpdir = aos::testing::TestTmpDir();
  const ::std::string base_name = tmpdir + "/logfile";
  const ::std::string config =
      absl::StrCat(base_name, kSingleConfigSha256, ".bfbs");
  const ::std::string logfile = base_name + "_data.part0.bfbs";
  // Remove it.
  unlink(config.c_str());
  unlink(logfile.c_str());

  LOG(INFO) << "Logging data to " << logfile;

  {
    event_loop_factory_.RunFor(chrono::milliseconds(95));

    std::unique_ptr<EventLoop> logger_event_loop =
        event_loop_factory_.MakeEventLoop("logger");
    Logger logger(logger_event_loop.get());
    logger.set_separate_config(false);
    logger.set_polling_period(std::chrono::milliseconds(100));
    logger.StartLoggingOnRun(base_name);
    event_loop_factory_.RunFor(chrono::milliseconds(20000));
  }

  // Even though it doesn't make any difference here, exercise the logic for
  // passing in a separate config.
  LogReader reader(logfile, &config_.message());

  const uint8_t *data_ptr = nullptr;
  reader.AddBeforeSendCallback<aos::examples::Ping>(
      "/test",
      [&data_ptr](aos::examples::Ping *ping,
                  const TimestampedMessage &timestamped_message) -> SharedSpan {
        ping->mutate_value(ping->value() + 10000);
        data_ptr = timestamped_message.data.get()->get()->data();
        return *timestamped_message.data;
      });

  // This sends out the fetched messages and advances time to the start of the
  // log file.
  reader.Register();

  EXPECT_THAT(reader.LoggedNodes(), ::testing::ElementsAre(nullptr));

  std::unique_ptr<EventLoop> test_event_loop =
      reader.event_loop_factory()->MakeEventLoop("log_reader");

  // Confirm that the ping and pong counts both match, and the value also
  // matches.
  int ping_count = 10010;
  test_event_loop->MakeWatcher(
      "/test",
      [&test_event_loop, &data_ptr, &ping_count](const examples::Ping &ping) {
        ++ping_count;
        EXPECT_EQ(ping.value(), ping_count);
        // Since simulated event loops (especially log replay) refcount the
        // shared data, we can verify if the right data got published by
        // verifying that the actual pointer to the flatbuffer matches.  This
        // only is guarenteed to hold during this callback.
        EXPECT_EQ(test_event_loop->context().data, data_ptr);
      });

  reader.event_loop_factory()->RunFor(std::chrono::seconds(100));
  EXPECT_EQ(ping_count, 12010);
}

// Tests calling StartLogging twice.
TEST_F(LoggerDeathTest, ExtraStart) {
  const ::std::string tmpdir = aos::testing::TestTmpDir();
  const ::std::string base_name1 = tmpdir + "/logfile1";
  const ::std::string config1 =
      absl::StrCat(base_name1, kSingleConfigSha256, ".bfbs");
  const ::std::string logfile1 = base_name1 + "_data.part0.bfbs";
  const ::std::string base_name2 = tmpdir + "/logfile2";
  const ::std::string config2 =
      absl::StrCat(base_name2, kSingleConfigSha256, ".bfbs");
  const ::std::string logfile2 = base_name2 + "_data.part0.bfbs";
  unlink(logfile1.c_str());
  unlink(config1.c_str());
  unlink(logfile2.c_str());
  unlink(config2.c_str());

  LOG(INFO) << "Logging data to " << logfile1 << " then " << logfile2;

  {
    event_loop_factory_.RunFor(chrono::milliseconds(95));

    std::unique_ptr<EventLoop> logger_event_loop =
        event_loop_factory_.MakeEventLoop("logger");
    Logger logger(logger_event_loop.get());
    logger.set_polling_period(std::chrono::milliseconds(100));
    logger_event_loop->OnRun([base_name1, base_name2, &logger_event_loop,
                              &logger]() {
      logger.StartLogging(std::make_unique<MultiNodeFilesLogNamer>(
          base_name1, logger_event_loop->configuration(),
          logger_event_loop.get(), logger_event_loop->node()));
      EXPECT_DEATH(logger.StartLogging(std::make_unique<MultiNodeFilesLogNamer>(
                       base_name2, logger_event_loop->configuration(),
                       logger_event_loop.get(), logger_event_loop->node())),
                   "Already logging");
    });
    event_loop_factory_.RunFor(chrono::milliseconds(20000));
  }
}

// Tests that we die if the replayer attempts to send on a logged channel.
TEST_F(LoggerDeathTest, DieOnDuplicateReplayChannels) {
  aos::FlatbufferDetachedBuffer<aos::Configuration> config =
      aos::configuration::ReadConfig(
          ArtifactPath("aos/events/pingpong_config.json"));
  SimulatedEventLoopFactory event_loop_factory(&config.message());
  const ::std::string tmpdir = aos::testing::TestTmpDir();
  const ::std::string base_name = tmpdir + "/logfile";
  const ::std::string config_file =
      absl::StrCat(base_name, kSingleConfigSha256, ".bfbs");
  const ::std::string logfile = base_name + "_data.part0.bfbs";
  // Remove the log file.
  unlink(config_file.c_str());
  unlink(logfile.c_str());

  LOG(INFO) << "Logging data to " << logfile;

  {
    std::unique_ptr<EventLoop> logger_event_loop =
        event_loop_factory.MakeEventLoop("logger");

    Logger logger(logger_event_loop.get());
    logger.set_separate_config(false);
    logger.set_polling_period(std::chrono::milliseconds(100));
    logger.StartLoggingOnRun(base_name);

    event_loop_factory.RunFor(chrono::seconds(2));
  }

  LogReader reader(logfile);

  reader.Register();

  std::unique_ptr<EventLoop> test_event_loop =
      reader.event_loop_factory()->MakeEventLoop("log_reader");

  EXPECT_DEATH(test_event_loop->MakeSender<examples::Ping>("/test"),
               "exclusive channel.*examples.Ping");
}

// Tests calling StopLogging twice.
TEST_F(LoggerDeathTest, ExtraStop) {
  const ::std::string tmpdir = aos::testing::TestTmpDir();
  const ::std::string base_name = tmpdir + "/logfile";
  const ::std::string config =
      absl::StrCat(base_name, kSingleConfigSha256, ".bfbs");
  const ::std::string logfile = base_name + ".part0.bfbs";
  // Remove it.
  unlink(config.c_str());
  unlink(logfile.c_str());

  LOG(INFO) << "Logging data to " << logfile;

  {
    event_loop_factory_.RunFor(chrono::milliseconds(95));

    std::unique_ptr<EventLoop> logger_event_loop =
        event_loop_factory_.MakeEventLoop("logger");
    Logger logger(logger_event_loop.get());
    logger.set_separate_config(false);
    logger.set_polling_period(std::chrono::milliseconds(100));
    logger_event_loop->OnRun([base_name, &logger_event_loop, &logger]() {
      logger.StartLogging(std::make_unique<MultiNodeFilesLogNamer>(
          base_name, logger_event_loop->configuration(),
          logger_event_loop.get(), logger_event_loop->node()));
      logger.StopLogging(aos::monotonic_clock::min_time);
      EXPECT_DEATH(logger.StopLogging(aos::monotonic_clock::min_time),
                   "Not logging right now");
    });
    event_loop_factory_.RunFor(chrono::milliseconds(20000));
  }
}

// Tests that we can startup twice.
TEST_F(LoggerTest, StartsTwice) {
  const ::std::string tmpdir = aos::testing::TestTmpDir();
  const ::std::string base_name1 = tmpdir + "/logfile1";
  const ::std::string config1 =
      absl::StrCat(base_name1, kSingleConfigSha256, ".bfbs");
  const ::std::string logfile1 = base_name1 + "_data.part0.bfbs";
  const ::std::string base_name2 = tmpdir + "/logfile2";
  const ::std::string config2 =
      absl::StrCat(base_name2, kSingleConfigSha256, ".bfbs");
  const ::std::string logfile2 = base_name2 + "_data.part0.bfbs";
  unlink(logfile1.c_str());
  unlink(config1.c_str());
  unlink(logfile2.c_str());
  unlink(config2.c_str());

  LOG(INFO) << "Logging data to " << logfile1 << " then " << logfile2;

  {
    event_loop_factory_.RunFor(chrono::milliseconds(95));

    std::unique_ptr<EventLoop> logger_event_loop =
        event_loop_factory_.MakeEventLoop("logger");
    Logger logger(logger_event_loop.get());
    logger.set_separate_config(false);
    logger.set_polling_period(std::chrono::milliseconds(100));
    logger.StartLogging(std::make_unique<MultiNodeFilesLogNamer>(
        base_name1, logger_event_loop->configuration(), logger_event_loop.get(),
        logger_event_loop->node()));
    event_loop_factory_.RunFor(chrono::milliseconds(10000));
    logger.StopLogging(logger_event_loop->monotonic_now());
    event_loop_factory_.RunFor(chrono::milliseconds(10000));
    logger.StartLogging(std::make_unique<MultiNodeFilesLogNamer>(
        base_name2, logger_event_loop->configuration(), logger_event_loop.get(),
        logger_event_loop->node()));
    event_loop_factory_.RunFor(chrono::milliseconds(10000));
  }

  for (const auto &logfile :
       {std::make_tuple(logfile1, 10), std::make_tuple(logfile2, 2010)}) {
    SCOPED_TRACE(std::get<0>(logfile));
    LogReader reader(std::get<0>(logfile));
    reader.Register();

    EXPECT_THAT(reader.LoggedNodes(), ::testing::ElementsAre(nullptr));

    std::unique_ptr<EventLoop> test_event_loop =
        reader.event_loop_factory()->MakeEventLoop("log_reader");

    int ping_count = std::get<1>(logfile);
    int pong_count = std::get<1>(logfile);

    // Confirm that the ping and pong counts both match, and the value also
    // matches.
    test_event_loop->MakeWatcher("/test",
                                 [&ping_count](const examples::Ping &ping) {
                                   EXPECT_EQ(ping.value(), ping_count + 1);
                                   ++ping_count;
                                 });
    test_event_loop->MakeWatcher(
        "/test", [&pong_count, &ping_count](const examples::Pong &pong) {
          EXPECT_EQ(pong.value(), pong_count + 1);
          ++pong_count;
          EXPECT_EQ(ping_count, pong_count);
        });

    reader.event_loop_factory()->RunFor(std::chrono::seconds(100));
    EXPECT_EQ(ping_count, std::get<1>(logfile) + 1000);
  }
}

// Tests that we can read and write rotated log files.
TEST_F(LoggerTest, RotatedLogFile) {
  const ::std::string tmpdir = aos::testing::TestTmpDir();
  const ::std::string base_name = tmpdir + "/logfile";
  const ::std::string config =
      absl::StrCat(base_name, kSingleConfigSha256, ".bfbs");
  const ::std::string logfile0 = base_name + "_data.part0.bfbs";
  const ::std::string logfile1 = base_name + "_data.part1.bfbs";
  // Remove it.
  unlink(config.c_str());
  unlink(logfile0.c_str());
  unlink(logfile1.c_str());

  LOG(INFO) << "Logging data to " << logfile0 << " and " << logfile1;

  {
    event_loop_factory_.RunFor(chrono::milliseconds(95));

    std::unique_ptr<EventLoop> logger_event_loop =
        event_loop_factory_.MakeEventLoop("logger");
    Logger logger(logger_event_loop.get());
    logger.set_separate_config(false);
    logger.set_polling_period(std::chrono::milliseconds(100));
    logger.StartLoggingOnRun(base_name);
    event_loop_factory_.RunFor(chrono::milliseconds(10000));
    logger.Rotate();
    event_loop_factory_.RunFor(chrono::milliseconds(10000));
  }

  {
    // Confirm that the UUIDs match for both the parts and the logger, and the
    // parts_index increments.
    std::vector<SizePrefixedFlatbufferVector<LogFileHeader>> log_header;
    for (std::string_view f : {logfile0, logfile1}) {
      log_header.emplace_back(ReadHeader(f).value());
    }

    EXPECT_EQ(log_header[0].message().log_event_uuid()->string_view(),
              log_header[1].message().log_event_uuid()->string_view());
    EXPECT_EQ(log_header[0].message().parts_uuid()->string_view(),
              log_header[1].message().parts_uuid()->string_view());

    EXPECT_EQ(log_header[0].message().parts_index(), 0);
    EXPECT_EQ(log_header[1].message().parts_index(), 1);
  }

  // Even though it doesn't make any difference here, exercise the logic for
  // passing in a separate config.
  LogReader reader(SortParts({logfile0, logfile1}), &config_.message());

  // Confirm that we can remap logged channels to point to new buses.
  reader.RemapLoggedChannel<aos::examples::Ping>("/test", "/original");

  // This sends out the fetched messages and advances time to the start of the
  // log file.
  reader.Register();

  EXPECT_THAT(reader.LoggedNodes(), ::testing::ElementsAre(nullptr));

  std::unique_ptr<EventLoop> test_event_loop =
      reader.event_loop_factory()->MakeEventLoop("log_reader");

  int ping_count = 10;
  int pong_count = 10;

  // Confirm that the ping value matches in the remapped channel location.
  test_event_loop->MakeWatcher("/original/test",
                               [&ping_count](const examples::Ping &ping) {
                                 EXPECT_EQ(ping.value(), ping_count + 1);
                                 ++ping_count;
                               });
  // Confirm that the ping and pong counts both match, and the value also
  // matches.
  test_event_loop->MakeWatcher(
      "/test", [&pong_count, &ping_count](const examples::Pong &pong) {
        EXPECT_EQ(pong.value(), pong_count + 1);
        ++pong_count;
        EXPECT_EQ(ping_count, pong_count);
      });

  reader.event_loop_factory()->RunFor(std::chrono::seconds(100));
  EXPECT_EQ(ping_count, 2010);
}

// Tests that a large number of messages per second doesn't overwhelm writev.
TEST_F(LoggerTest, ManyMessages) {
  const ::std::string tmpdir = aos::testing::TestTmpDir();
  const ::std::string base_name = tmpdir + "/logfile";
  const ::std::string config =
      absl::StrCat(base_name, kSingleConfigSha256, ".bfbs");
  const ::std::string logfile = base_name + ".part0.bfbs";
  // Remove the log file.
  unlink(config.c_str());
  unlink(logfile.c_str());

  LOG(INFO) << "Logging data to " << logfile;
  ping_.set_quiet(true);

  {
    std::unique_ptr<EventLoop> logger_event_loop =
        event_loop_factory_.MakeEventLoop("logger");

    std::unique_ptr<EventLoop> ping_spammer_event_loop =
        event_loop_factory_.MakeEventLoop("ping_spammer");
    aos::Sender<examples::Ping> ping_sender =
        ping_spammer_event_loop->MakeSender<examples::Ping>("/test");

    aos::TimerHandler *timer_handler =
        ping_spammer_event_loop->AddTimer([&ping_sender]() {
          aos::Sender<examples::Ping>::Builder builder =
              ping_sender.MakeBuilder();
          examples::Ping::Builder ping_builder =
              builder.MakeBuilder<examples::Ping>();
          CHECK_EQ(builder.Send(ping_builder.Finish()), RawSender::Error::kOk);
        });

    // 100 ms / 0.05 ms -> 2000 messages.  Should be enough to crash it.
    ping_spammer_event_loop->OnRun([&ping_spammer_event_loop, timer_handler]() {
      timer_handler->Schedule(ping_spammer_event_loop->monotonic_now(),
                              chrono::microseconds(50));
    });

    Logger logger(logger_event_loop.get());
    logger.set_separate_config(false);
    logger.set_polling_period(std::chrono::milliseconds(100));
    logger.StartLoggingOnRun(base_name);

    event_loop_factory_.RunFor(chrono::milliseconds(1000));
  }
}

// Tests that we can read a logfile that has channels which were sent too fast.
TEST(SingleNodeLoggerNoFixtureTest, ReadTooFast) {
  aos::FlatbufferDetachedBuffer<aos::Configuration> config =
      aos::configuration::ReadConfig(
          ArtifactPath("aos/events/pingpong_config.json"));
  SimulatedEventLoopFactory event_loop_factory(&config.message());
  const ::std::string tmpdir = aos::testing::TestTmpDir();
  const ::std::string base_name = tmpdir + "/logfile";
  const ::std::string config_file =
      absl::StrCat(base_name, kSingleConfigSha256, ".bfbs");
  const ::std::string logfile = base_name + "_data.part0.bfbs";
  // Remove the log file.
  unlink(config_file.c_str());
  unlink(logfile.c_str());

  LOG(INFO) << "Logging data to " << logfile;

  int sent_messages = 0;

  {
    std::unique_ptr<EventLoop> logger_event_loop =
        event_loop_factory.MakeEventLoop("logger");

    std::unique_ptr<EventLoop> ping_spammer_event_loop =
        event_loop_factory.GetNodeEventLoopFactory(nullptr)->MakeEventLoop(
            "ping_spammer", {NodeEventLoopFactory::CheckSentTooFast::kNo,
                             NodeEventLoopFactory::ExclusiveSenders::kNo,
                             {}});
    aos::Sender<examples::Ping> ping_sender =
        ping_spammer_event_loop->MakeSender<examples::Ping>("/test");

    aos::TimerHandler *timer_handler =
        ping_spammer_event_loop->AddTimer([&ping_sender, &sent_messages]() {
          aos::Sender<examples::Ping>::Builder builder =
              ping_sender.MakeBuilder();
          examples::Ping::Builder ping_builder =
              builder.MakeBuilder<examples::Ping>();
          CHECK_EQ(builder.Send(ping_builder.Finish()), RawSender::Error::kOk);
          ++sent_messages;
        });

    constexpr std::chrono::microseconds kSendPeriod{10};
    const int max_legal_messages = configuration::QueueSize(
        event_loop_factory.configuration(), ping_sender.channel());

    ping_spammer_event_loop->OnRun(
        [&ping_spammer_event_loop, kSendPeriod, timer_handler]() {
          timer_handler->Schedule(
              ping_spammer_event_loop->monotonic_now() + kSendPeriod / 2,
              kSendPeriod);
        });

    Logger logger(logger_event_loop.get());
    logger.set_separate_config(false);
    logger.set_polling_period(std::chrono::milliseconds(100));
    logger.StartLoggingOnRun(base_name);

    event_loop_factory.RunFor(kSendPeriod * max_legal_messages * 2);
  }

  LogReader reader(logfile);

  reader.Register();

  std::unique_ptr<EventLoop> test_event_loop =
      reader.event_loop_factory()->MakeEventLoop("log_reader");

  int replay_count = 0;

  test_event_loop->MakeWatcher(
      "/test", [&replay_count](const examples::Ping &) { ++replay_count; });

  reader.event_loop_factory()->Run();
  EXPECT_EQ(replay_count, sent_messages);
}

// Helper function to verify the contents of the profiling data file.
void VerifyProfilingData(const std::filesystem::path &profiling_path) {
  std::ifstream file(profiling_path);
  ASSERT_TRUE(file.is_open()) << "Failed to open profiling data file.";

  std::string line;

  // Verify that the header is a comment starting with '#'.
  std::getline(file, line);
  ASSERT_THAT(line, ::testing::StartsWith("#"));

  // Now, verify the contents of each line.
  int record_count = 0;

  // Track the total encoding duration.
  uint64_t total_encoding_duration_ns = 0;
  std::vector<int64_t> encode_durations_ns;

  while (std::getline(file, line)) {
    std::stringstream line_stream(line);
    std::string cell;

    // Extract each cell from the CSV line.
    std::vector<std::string> cells;
    while (std::getline(line_stream, cell, ',')) {
      cells.push_back(cell);
    }

    // Expecting 5 fields: channel_name, channel_type, encode_duration_ns,
    // encoding_start_time_ns, message_time_s.
    ASSERT_EQ(cells.size(), 5) << "Incorrect number of fields in the CSV line.";

    // Channel name and type are strings and just need to not be empty.
    EXPECT_FALSE(cells[0].empty()) << "Channel name is empty.";
    EXPECT_FALSE(cells[1].empty()) << "Channel type is empty.";

    // Encode duration, encoding start time should be positive numbers.
    const int64_t encode_duration_ns = std::stoll(cells[2]);
    const int64_t encoding_start_time_ns = std::stoll(cells[3]);

    encode_durations_ns.push_back(encode_duration_ns);

    ASSERT_GT(encode_duration_ns, 0)
        << "Encode duration is not positive. Line: " << line;
    ASSERT_GT(encoding_start_time_ns, 0)
        << "Encoding start time is not positive. Line: " << line;

    // Message time should be non-negative.
    const double message_time_s = std::stod(cells[4]);
    EXPECT_GE(message_time_s, 0) << "Message time is negative";
    ++record_count;
    total_encoding_duration_ns += encode_duration_ns;
  }

  EXPECT_GT(record_count, 0) << "Profiling data file is empty.";
  LOG(INFO) << "Total encoding duration: " << total_encoding_duration_ns;

  std::sort(encode_durations_ns.begin(), encode_durations_ns.end());

  // calculate the minimum encode duration.
  const int64_t min_encode_duration_ns = encode_durations_ns.front();
  LOG(INFO) << "Minimum encoding duration: " << min_encode_duration_ns;

  // calculate the maximum encode duration.
  const int64_t max_encode_duration_ns = encode_durations_ns.back();
  LOG(INFO) << "Maximum encoding duration: " << max_encode_duration_ns;

  // calculate the median encode duration.
  const int median_index = encode_durations_ns.size() / 2;
  const int64_t median_encode_duration_ns = encode_durations_ns[median_index];
  LOG(INFO) << "Median encoding duration: " << median_encode_duration_ns;

  // calculate the average encode duration.
  const int64_t average_encode_duration_ns =
      total_encoding_duration_ns / record_count;
  LOG(INFO) << "Average encoding duration: " << average_encode_duration_ns;
}

// Tests logging many messages with LZMA compression.
TEST_F(LoggerTest, ManyMessagesLzmaWithProfiling) {
  const std::string tmpdir = aos::testing::TestTmpDir();
  const std::string base_name = tmpdir + "/lzma_logfile";
  const std::string config_sha256 =
      absl::StrCat(base_name, kSingleConfigSha256, ".bfbs");
  const std::string logfile = absl::StrCat(base_name, ".part0.xz");
  const std::string profiling_path =
      absl::StrCat(tmpdir, "/encoding_profile.csv");

  // Clean up any previous test artifacts.
  unlink(config_sha256.c_str());
  unlink(logfile.c_str());

  LOG(INFO) << "Logging data to " << logfile;
  ping_.set_quiet(true);

  {
    std::unique_ptr<aos::EventLoop> logger_event_loop =
        event_loop_factory_.MakeEventLoop("logger");

    std::unique_ptr<aos::EventLoop> ping_spammer_event_loop =
        event_loop_factory_.MakeEventLoop("ping_spammer");
    aos::Sender<examples::Ping> ping_sender =
        ping_spammer_event_loop->MakeSender<examples::Ping>("/test");

    aos::TimerHandler *timer_handler =
        ping_spammer_event_loop->AddTimer([&ping_sender]() {
          aos::Sender<examples::Ping>::Builder builder =
              ping_sender.MakeBuilder();
          examples::Ping::Builder ping_builder =
              builder.MakeBuilder<examples::Ping>();
          CHECK_EQ(builder.Send(ping_builder.Finish()),
                   aos::RawSender::Error::kOk);
        });

    // Send a message every 50 microseconds to simulate high throughput.
    ping_spammer_event_loop->OnRun([&ping_spammer_event_loop, timer_handler]() {
      timer_handler->Schedule(ping_spammer_event_loop->monotonic_now(),
                              std::chrono::microseconds(50));
    });

    aos::logger::Logger logger(logger_event_loop.get());
    logger.set_separate_config(false);
    logger.set_polling_period(std::chrono::milliseconds(100));

    // Enable logger profiling.
    logger.SetProfilingPath(profiling_path);

    std::unique_ptr<aos::logger::MultiNodeFilesLogNamer> log_namer =
        std::make_unique<aos::logger::MultiNodeFilesLogNamer>(
            base_name, logger_event_loop->configuration(),
            logger_event_loop.get(), logger_event_loop->node());
#ifdef LZMA
    // Set up LZMA encoder.
    log_namer->set_encoder_factory([](size_t max_message_size) {
      return std::make_unique<aos::logger::LzmaEncoder>(max_message_size, 1);
    });
#endif

    logger.StartLogging(std::move(log_namer));

    event_loop_factory_.RunFor(std::chrono::seconds(1));
  }

#ifdef LZMA
  VerifyProfilingData(profiling_path);
#endif
}

}  // namespace aos::logger::testing
