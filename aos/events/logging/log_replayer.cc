#include <stdlib.h>

#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "flatbuffers/flatbuffers.h"

#include "aos/configuration_generated.h"
#include "aos/events/event_loop.h"
#include "aos/events/logging/log_reader.h"
#include "aos/events/logging/log_reader_utils.h"
#include "aos/events/logging/log_replayer_config_generated.h"
#include "aos/events/logging/log_replayer_stats_generated.h"
#include "aos/events/logging/log_replayer_stats_schema.h"
#include "aos/events/logging/logfile_sorting.h"
#include "aos/events/logging/logfile_utils.h"
#include "aos/events/logging/replay_timing_generated.h"
#include "aos/events/logging/replay_timing_schema.h"
#include "aos/events/shm_event_loop.h"
#include "aos/flatbuffer_merge.h"
#include "aos/init.h"
#include "aos/json_to_flatbuffer.h"
#include "aos/util/file.h"

ABSL_FLAG(std::string, config, "",
          "If specified, overrides logged configuration.");
ABSL_FLAG(
    bool, plot_timing, true,
    "If set, generates a plot of the replay timing--namely, the errors between "
    "when we "
    "should've sent messages and when we actually sent replayed messages.");
ABSL_FLAG(bool, skip_sender_channels, true,
          "If set, skips replay of the channels applications replay on");
ABSL_FLAG(bool, skip_replay, false,
          "If set, skips actually running the replay. Useful for writing a "
          "config without running replay");
ABSL_FLAG(
    bool, print_config, false,
    "If set, prints the config that will be used for replay to stdout as json");
ABSL_FLAG(
    std::string, replay_config, "",
    "Path to the configuration used for log replay which includes items such "
    "as channels to remap, and applications to target for replay. If not set, "
    "log_reader will run on shm event loop. ");
ABSL_FLAG(std::string, merge_with_config, "",
          "A valid json string to be merged with config. This is used to "
          "add extra applications needed to run only for log_replayer");
ABSL_FLAG(bool, print_stats, true,
          "if set, prints the LogReplayerStats message as JSON to stdout");

namespace aos::logger {

int Main(int argc, char *argv[]) {
  const std::vector<aos::logger::LogFile> logfiles =
      aos::logger::SortParts(aos::logger::FindLogs(argc, argv));

  aos::logger::LogReader config_reader(logfiles);
  aos::FlatbufferDetachedBuffer<aos::Configuration> config =
      absl::GetFlag(FLAGS_config).empty()
          ? CopyFlatBuffer<aos::Configuration>(config_reader.configuration())
          : aos::configuration::ReadConfig(absl::GetFlag(FLAGS_config));

  if (absl::GetFlag(FLAGS_plot_timing)) {
    // Go through the effort to add a ReplayTiming channel to ensure that we
    // can capture timing information from the replay.
    const aos::Configuration *raw_config = &config.message();
    aos::ChannelT channel_overrides;
    channel_overrides.max_size = 10000;
    channel_overrides.frequency = 10000;
    config = aos::configuration::AddChannelToConfiguration(
        raw_config, "/timing",
        aos::FlatbufferSpan<reflection::Schema>(
            aos::timing::ReplayTimingSchema()),
        aos::configuration::GetMyNode(raw_config), channel_overrides);
  }

  // Add the LogReplayerStats channel
  const aos::Configuration *raw_config = &config.message();
  aos::ChannelT channel_overrides;
  channel_overrides.max_size = 10000;
  channel_overrides.frequency = 1;
  config = aos::configuration::AddChannelToConfiguration(
      raw_config, "/replay",
      aos::FlatbufferSpan<reflection::Schema>(aos::LogReplayerStatsSchema()),
      aos::configuration::GetMyNode(raw_config), channel_overrides);

  if (!absl::GetFlag(FLAGS_merge_with_config).empty()) {
    config = aos::configuration::MergeWithConfig(
        &config.message(), absl::GetFlag(FLAGS_merge_with_config));
  }

  std::optional<aos::FlatbufferDetachedBuffer<ReplayConfig>> replay_config =
      absl::GetFlag(FLAGS_replay_config).empty()
          ? std::nullopt
          : std::make_optional(aos::JsonToFlatbuffer<ReplayConfig>(
                aos::util::ReadFileToStringOrDie(
                    absl::GetFlag(FLAGS_replay_config).data())));
  std::vector<std::pair<std::string, std::string>> message_filter;
  if (absl::GetFlag(FLAGS_skip_sender_channels) && replay_config.has_value()) {
    CHECK(replay_config.value().message().has_active_nodes());
    std::vector<const Node *> active_nodes;
    for (const auto &node : *replay_config.value().message().active_nodes()) {
      active_nodes.emplace_back(configuration::GetNode(
          &config.message(), node->name()->string_view()));
    }

    std::vector<std::string> applications;
    for (const auto &application :
         *replay_config.value().message().applications()) {
      if (application->name()->string_view() != "camera_message_interceptor") {
        applications.emplace_back(application->name()->string_view());
      }
    }

    aos::logger::ChannelsInLogResult channels =
        ChannelsInLog(logfiles, active_nodes, applications);
    for (auto const &channel :
         channels.watchers_and_fetchers_without_senders.value()) {
      message_filter.emplace_back(channel.name, channel.type);
    }
  }

  aos::logger::LogReader reader(
      logfiles, &config.message(),
      message_filter.empty() ? nullptr : &message_filter);

  if (replay_config.has_value() &&
      replay_config.value().message().has_remap_channels()) {
    for (auto const &remap_channel :
         *replay_config.value().message().remap_channels()) {
      auto const &channel = remap_channel->channel();
      std::string_view new_type = remap_channel->has_new_type()
                                      ? remap_channel->new_type()->string_view()
                                      : channel->type()->string_view();
      reader.RemapLoggedChannel(
          channel->name()->string_view(), channel->type()->string_view(),
          remap_channel->prefix()->string_view(), new_type);
    }
  }

  if (absl::GetFlag(FLAGS_print_config)) {
    // TODO(Naman): Replace with config writer if it will be cleaner
    std::cout << FlatbufferToJson(reader.configuration()) << std::endl;
  }

  if (!absl::GetFlag(FLAGS_skip_replay)) {
    aos::ShmEventLoop event_loop(reader.configuration());

    event_loop.SkipAosLog();
    event_loop.SkipTimingReport();

    aos::Sender<aos::LogReplayerStats> stats_sender =
        event_loop.MakeSender<aos::LogReplayerStats>("/replay");
    auto builder = stats_sender.MakeBuilder();
    auto node_name = builder.fbb()->CreateString(event_loop.node()->name());
    flatbuffers::Offset<aos::ReplayConfig> replay_config_offset;
    if (replay_config.has_value()) {
      replay_config_offset =
          aos::CopyFlatBuffer(&replay_config.value().message(), builder.fbb());
    }

    auto stats_builder = builder.MakeBuilder<aos::LogReplayerStats>();
    if (replay_config.has_value()) {
      stats_builder.add_replay_config(replay_config_offset);
    }

    reader.Register(&event_loop);

    // Save off the start and end times of replay.
    reader.OnStart(event_loop.node(), [&event_loop, &stats_builder,
                                       &node_name]() {
      stats_builder.add_node(node_name);
      stats_builder.add_realtime_start_time(
          std::chrono::nanoseconds(event_loop.realtime_now().time_since_epoch())
              .count());

      stats_builder.add_monotonic_start_time(
          std::chrono::nanoseconds(
              event_loop.monotonic_now().time_since_epoch())
              .count());
    });

    reader.OnEnd(event_loop.node(), [&event_loop, &stats_builder, &builder]() {
      stats_builder.add_realtime_end_time(
          std::chrono::nanoseconds(event_loop.realtime_now().time_since_epoch())
              .count());

      stats_builder.add_monotonic_end_time(
          std::chrono::nanoseconds(
              event_loop.monotonic_now().time_since_epoch())
              .count());
      builder.CheckOk(builder.Send(stats_builder.Finish()));
    });

    reader.OnEnd(event_loop.node(), [&event_loop]() { event_loop.Exit(); });

    if (absl::GetFlag(FLAGS_plot_timing)) {
      aos::Sender<aos::timing::ReplayTiming> replay_timing_sender =
          event_loop.MakeSender<aos::timing::ReplayTiming>("/timing");
      reader.set_timing_accuracy_sender(event_loop.node(),
                                        std::move(replay_timing_sender));
    }

    event_loop.Run();

    reader.Deregister();

    if (absl::GetFlag(FLAGS_print_stats)) {
      aos::Fetcher<aos::LogReplayerStats> stats_fetcher =
          event_loop.MakeFetcher<aos::LogReplayerStats>("/replay");
      CHECK(stats_fetcher.Fetch()) << "Failed to fetch LogReplayerStats!";
      std::cout << aos::FlatbufferToJson(stats_fetcher.get());
    }
  }

  return EXIT_SUCCESS;
}

}  // namespace aos::logger

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage(
      R"message(Binary to replay the full contents of a logfile into shared memory.
                #replay_config should be set in order to replay a set of nodes, applications and channels
                #print config and skip replay, if you only want to print the config and not do log replay
                Use case #1: log_replayer <log_dir> --print_config --replay_config=<path_to_config> --skip_replay
                Use case #2: log_replayer <log_dir> --nofatal_sent_too_fast --replay_config=<path_to_config>
      )message");

  aos::InitGoogle(&argc, &argv);
  return aos::logger::Main(argc, argv);
}
