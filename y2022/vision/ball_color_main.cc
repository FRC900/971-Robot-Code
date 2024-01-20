#include "aos/events/shm_event_loop.h"
#include "aos/init.h"
#include "y2022/constants.h"
#include "y2022/vision/ball_color.h"

// config used to allow running ball_color_detector independently.  E.g.,
// bazel run //y2022/vision:ball_color_detector -- --config
// y2022/aos_config.json
//   --override_hostname pi-7971-1  --ignore_timestamps true
DEFINE_string(config, "aos_config.json", "Path to the config file to use.");

namespace y2022::vision {
namespace {

using namespace frc971::vision;

void BallColorDetectorMain() {
  aos::FlatbufferDetachedBuffer<aos::Configuration> config =
      aos::configuration::ReadConfig(FLAGS_config);

  aos::ShmEventLoop event_loop(&config.message());
  std::shared_ptr<const constants::Values> values =
      std::make_shared<const constants::Values>(constants::MakeValues());
  BallColorDetector ball_color_detector(&event_loop, std::move(values));

  event_loop.Run();
}

}  // namespace
}  // namespace y2022::vision

int main(int argc, char **argv) {
  aos::InitGoogle(&argc, &argv);
  y2022::vision::BallColorDetectorMain();
}
