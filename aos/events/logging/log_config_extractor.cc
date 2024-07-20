#include <filesystem>
#include <iostream>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "flatbuffers/flatbuffers.h"

#include "aos/configuration_generated.h"
#include "aos/events/logging/log_reader.h"
#include "aos/events/logging/logfile_sorting.h"
#include "aos/flatbuffer_merge.h"
#include "aos/init.h"
#include "aos/json_to_flatbuffer.h"

ABSL_FLAG(std::string, output_path, "/tmp/",
          "Destination folder for output files. If this flag is not used, "
          "it stores the files in /tmp/.");
ABSL_FLAG(bool, convert_to_json, false,
          "If true, can be used to convert bfbs to json.");
ABSL_FLAG(bool, bfbs, false,
          "If true, write as a binary flatbuffer inside the output_path.");
ABSL_FLAG(bool, json, false,
          "If true, write as a json inside the output_path.");
ABSL_FLAG(bool, stripped, false,
          "If true, write as a stripped json inside the output_path.");
ABSL_FLAG(bool, quiet, false,
          "If true, do not print configuration to stdout. If false, print "
          "stripped json");

namespace aos {

void WriteConfig(const aos::Configuration *config, std::string output_path) {
  // In order to deduplicate the reflection schemas in the resulting config
  // flatbuffer (and thus reduce the size of the final flatbuffer), use
  // MergeConfiguration() rather than the more generic
  // RecursiveCopyFlatBuffer(). RecursiveCopyFlatBuffer() is sometimes used to
  // reduce flatbuffer memory usage, but it only does so by rearranging memory
  // locations, not by actually deduplicating identical flatbuffers.
  std::vector<aos::FlatbufferVector<reflection::Schema>> schemas;
  for (const Channel *c : *config->channels()) {
    schemas.emplace_back(RecursiveCopyFlatBuffer(c->schema()));
  }
  auto config_flatbuffer = configuration::MergeConfiguration(
      RecursiveCopyFlatBuffer(config), schemas);

  if (absl::GetFlag(FLAGS_bfbs)) {
    WriteFlatbufferToFile(output_path + ".bfbs", config_flatbuffer);
    LOG(INFO) << "Done writing bfbs to " << output_path << ".bfbs";
  }

  if (absl::GetFlag(FLAGS_json)) {
    WriteFlatbufferToJson(output_path + ".json", config_flatbuffer);
    LOG(INFO) << "Done writing json to " << output_path << ".json";
  }

  if (absl::GetFlag(FLAGS_stripped) || !absl::GetFlag(FLAGS_quiet)) {
    auto *channels = config_flatbuffer.mutable_message()->mutable_channels();
    for (size_t i = 0; i < channels->size(); i++) {
      channels->GetMutableObject(i)->clear_schema();
    }
    if (absl::GetFlag(FLAGS_stripped)) {
      WriteFlatbufferToJson(output_path + ".stripped.json", config_flatbuffer);
      LOG(INFO) << "Done writing stripped json to " << output_path
                << ".stripped.json";
    }
    if (!absl::GetFlag(FLAGS_quiet)) {
      std::cout << FlatbufferToJson(config_flatbuffer) << std::endl;
    }
  }
}

int Main(int argc, char *argv[]) {
  CHECK(argc > 1) << "Must provide an argument";

  std::string output_path = absl::GetFlag(FLAGS_output_path);
  if (output_path.back() != '/') {
    output_path += "/";
  }
  if (!std::filesystem::exists(output_path)) {
    LOG(ERROR)
        << "Output path is invalid. Make sure the path exists before running.";
    return EXIT_FAILURE;
  }
  output_path += "aos_config";

  std::shared_ptr<const aos::Configuration> config;
  // Check if the user wants to use stdin (denoted by '-') which will help
  // convert configs in json to bfbs (see example in SetUsageMessage)
  std::string_view arg{argv[1]};
  if (arg == "-") {
    // Read in everything from stdin, blocks when there's no data on stdin
    std::string stdin_data(std::istreambuf_iterator(std::cin), {});
    aos::FlatbufferDetachedBuffer<aos::Configuration> buffer(
        aos::JsonToFlatbuffer(stdin_data, aos::ConfigurationTypeTable()));
    WriteConfig(&buffer.message(), output_path);
  } else if (absl::GetFlag(FLAGS_convert_to_json)) {
    aos::FlatbufferDetachedBuffer config = aos::configuration::ReadConfig(arg);
    WriteFlatbufferToJson(output_path + ".json", config);
    LOG(INFO) << "Done writing json to " << output_path << ".json";
  } else {
    const std::vector<aos::logger::LogFile> logfiles =
        aos::logger::SortParts(aos::logger::FindLogs(argc, argv));

    WriteConfig(logfiles[0].config.get(), output_path);
  }
  return EXIT_SUCCESS;
}

}  // namespace aos

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage(
      "Binary to output the configuration of a log.\n"
      "# print config as stripped json to stdout\n"
      "# path to log should always be absolute path.\n"
      "log_config_extractor /path/to/log\n"
      "# write config to ~/work/results/aos_config.bfbs and "
      "~/work/results/aos_config.json with no stdout "
      "output\n"
      "# make sure the output paths are valid and absolute paths.\n"
      "log_config_extractor /path/to/log --output_path=~/work/results/ --bfbs "
      "--json --quiet\n"
      "# pass json config by stdin and output as bfbs\n"
      "cat aos_config.json | log_config_extractor - --output_path "
      "/absolute/path/to/dir --bfbs\n"
      "# This can also be used to convert a bfbs file of config to json\n"
      "log_config_extractor /path/to/config.bfbs --convert_to_json");

  aos::InitGoogle(&argc, &argv);
  return aos::Main(argc, argv);
}
