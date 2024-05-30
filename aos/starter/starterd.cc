#include <pwd.h>
#include <unistd.h>

#include <ostream>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "aos/configuration.h"
#include "aos/events/event_loop.h"
#include "aos/flatbuffers.h"
#include "aos/init.h"
#include "aos/starter/starterd_lib.h"
#include "aos/util/file.h"

DEFINE_string(config, "aos_config.json", "File path of aos configuration");
DEFINE_string(user, "",
              "Starter runs as though this user ran a SUID binary if set.");
DEFINE_string(version_string, "",
              "Version to report for starterd and subprocesses.");

DECLARE_string(shm_base);
DEFINE_bool(purge_shm_base, false,
            "If true, delete everything in --shm_base before starting.");

int main(int argc, char **argv) {
  aos::InitGoogle(&argc, &argv);

  if (FLAGS_purge_shm_base) {
    aos::util::UnlinkRecursive(FLAGS_shm_base);
  }

  if (!FLAGS_user.empty()) {
    uid_t uid;
    uid_t gid;
    {
      struct passwd *user_data = getpwnam(FLAGS_user.c_str());
      if (user_data != nullptr) {
        uid = user_data->pw_uid;
        gid = user_data->pw_gid;
      } else {
        LOG(FATAL) << "Could not find user " << FLAGS_user;
        return 1;
      }
    }
    // Change the real and effective IDs to the user we're running as. The
    // effective IDs mean files we access (like shared memory) will happen as
    // that user. The real IDs allow child processes with an different effective
    // ID to still participate in signal sending/receiving.
    constexpr int kUnchanged = -1;
    if (setresgid(/* ruid */ gid, /* euid */ gid,
                  /* suid */ kUnchanged) != 0) {
      PLOG(FATAL) << "Failed to change GID to " << FLAGS_user << ", group "
                  << gid;
    }

    if (setresuid(/* ruid */ uid, /* euid */ uid,
                  /* suid */ kUnchanged) != 0) {
      PLOG(FATAL) << "Failed to change UID to " << FLAGS_user;
    }
  }

  aos::FlatbufferDetachedBuffer<aos::Configuration> config =
      aos::configuration::ReadConfig(FLAGS_config);

  const aos::Configuration *config_msg = &config.message();

  aos::starter::Starter starter(config_msg);
  if (!FLAGS_version_string.empty()) {
    starter.event_loop()->SetVersionString(FLAGS_version_string);
  }

  starter.Run();

  return 0;
}
