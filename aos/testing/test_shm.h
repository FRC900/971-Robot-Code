#ifndef AOS_TESTING_TEST_SHM_H_
#define AOS_TESTING_TEST_SHM_H_

#include "aos/ipc_lib/shared_mem_types.h"

namespace aos::testing {

// Manages creating and cleaning up "shared memory" which works within this
// process and any that it fork(2)s.
class TestSharedMemory {
 public:
  // Calls EnableTestLogging().
  TestSharedMemory();
  ~TestSharedMemory();

 private:
  struct aos_core global_core_data_;
};

}  // namespace aos::testing

#endif  // AOS_TESTING_TEST_SHM_H_
