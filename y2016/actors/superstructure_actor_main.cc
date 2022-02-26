#include <cstdio>

#include "aos/events/shm_event_loop.h"
#include "aos/init.h"
#include "y2016/actors/superstructure_actor.h"

int main(int argc, char *argv[]) {
  ::aos::InitGoogle(&argc, &argv);

  aos::FlatbufferDetachedBuffer<aos::Configuration> config =
      aos::configuration::ReadConfig("aos_config.json");

  ::aos::ShmEventLoop event_loop(&config.message());
  ::y2016::actors::SuperstructureActor superstructure(&event_loop);
  event_loop.Run();

  return 0;
}
