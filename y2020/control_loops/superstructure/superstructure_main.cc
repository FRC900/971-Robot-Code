#include "y2020/constants.h"
#include "y2020/control_loops/superstructure/superstructure.h"

#include "aos/events/shm_event_loop.h"
#include "aos/init.h"

int main(int argc, char **argv) {
  ::aos::InitGoogle(&argc, &argv);

  aos::FlatbufferDetachedBuffer<aos::Configuration> config =
      aos::configuration::ReadConfig("aos_config.json");

  ::aos::ShmEventLoop event_loop(&config.message());
  ::y2020::constants::InitValues();
  ::y2020::control_loops::superstructure::Superstructure superstructure(
      &event_loop);

  event_loop.Run();

  return 0;
}
