#include "aos/ipc_lib/shared_mem.h"

#include <fcntl.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ostream>

#include "absl/log/check.h"
#include "absl/log/log.h"

#include "aos/ipc_lib/aos_sync.h"

// the path for the shared memory segment. see shm_open(3) for restrictions
#define AOS_SHM_NAME "/aos_shared_mem"
// Size of the shared mem segment.
// This must fit in the tmpfs for /dev/shm/
#define SIZEOFSHMSEG (4096 * 0x800)

void init_shared_mem_core(aos_shm_core *shm_core) {
  memset(&shm_core->time_offset, 0, sizeof(shm_core->time_offset));
  memset(&shm_core->msg_alloc_lock, 0, sizeof(shm_core->msg_alloc_lock));
  shm_core->queues.pointer = NULL;
  memset(&shm_core->queues.lock, 0, sizeof(shm_core->queues.lock));
  shm_core->queue_types.pointer = NULL;
  memset(&shm_core->queue_types.lock, 0, sizeof(shm_core->queue_types.lock));
}

ptrdiff_t aos_core_get_mem_usage(void) {
  return global_core->size - ((ptrdiff_t)global_core->mem_struct->msg_alloc -
                              (ptrdiff_t)global_core->mem_struct);
}

struct aos_core *global_core = NULL;

// TODO(brians): madvise(2) it to put this shm in core dumps.
void aos_core_create_shared_mem(int create, int lock) {
  assert(global_core == NULL);
  static struct aos_core global_core_data;
  global_core = &global_core_data;

  {
    char *shm_name = getenv("AOS_SHM_NAME");
    if (shm_name == NULL) {
      global_core->shm_name = AOS_SHM_NAME;
    } else {
      printf("AOS_SHM_NAME defined, using %s\n", shm_name);
      global_core->shm_name = shm_name;
    }
  }

  int shm;
  if (create) {
    while (1) {
      shm = shm_open(global_core->shm_name, O_RDWR | O_CREAT | O_EXCL, 0666);
      global_core->owner = 1;
      if (shm == -1 && errno == EEXIST) {
        printf("shared_mem: going to shm_unlink(%s)\n", global_core->shm_name);
        if (shm_unlink(global_core->shm_name) == -1) {
          PLOG(WARNING) << "shm_unlink(" << global_core->shm_name << ") failed";
          break;
        }
      } else {
        break;
      }
    }
  } else {
    shm = shm_open(global_core->shm_name, O_RDWR, 0);
    global_core->owner = 0;
  }
  if (shm == -1) {
    PLOG(FATAL) << "shm_open(" << global_core->shm_name
                << ", O_RDWR [| O_CREAT | O_EXCL, 0|0666) failed";
  }
  if (global_core->owner) {
    PCHECK(ftruncate(shm, SIZEOFSHMSEG) == 0)
        << ": fruncate(" << shm << ", 0x" << std::hex << (size_t)SIZEOFSHMSEG
        << ") failed";
  }
  int flags = MAP_SHARED | MAP_FIXED;
  if (lock) flags |= MAP_LOCKED | MAP_POPULATE;
  void *shm_address = mmap((void *)SHM_START, SIZEOFSHMSEG,
                           PROT_READ | PROT_WRITE, flags, shm, 0);
  PCHECK(shm_address != MAP_FAILED)
      << std::hex << "shared_mem: mmap(" << (void *)SHM_START << ", 0x"
      << (size_t)SIZEOFSHMSEG << ", stuff, " << flags << ", " << shm
      << ", 0) failed";
  if (create) {
    printf("shared_mem: creating %s, shm at: %p\n", global_core->shm_name,
           shm_address);
  } else {
    printf("shared_mem: not creating, shm at: %p\n", shm_address);
  }
  if (close(shm) == -1) {
    PLOG(WARNING) << "close(" << shm << "(=shm) failed";
  }
  PCHECK(shm_address == (void *)SHM_START)
      << "shm isn't at hard-coded " << (void *)SHM_START << ". at "
      << shm_address << " instead";
  aos_core_use_address_as_shared_mem(shm_address, SIZEOFSHMSEG);
  LOG(INFO) << "shared_mem: end of create_shared_mem owner="
            << global_core->owner;
}

void aos_core_use_address_as_shared_mem(void *address, size_t size) {
  global_core->mem_struct = reinterpret_cast<aos_shm_core_t *>(address);
  global_core->size = size;
  global_core->shared_mem =
      (uint8_t *)address + sizeof(*global_core->mem_struct);
  if (global_core->owner) {
    global_core->mem_struct->msg_alloc = (uint8_t *)address + global_core->size;
    init_shared_mem_core(global_core->mem_struct);
    futex_set(&global_core->mem_struct->creation_condition);
  } else {
    if (futex_wait(&global_core->mem_struct->creation_condition) != 0) {
      LOG(FATAL) << "waiting on creation_condition failed";
    }
  }
}

void aos_core_free_shared_mem() {
  if (global_core != nullptr) {
    void *shm_address = global_core->shared_mem;
    PCHECK(munmap((void *)SHM_START, SIZEOFSHMSEG) != -1)
        << ": munmap(" << shm_address << ", 0x" << std::hex
        << (size_t)SIZEOFSHMSEG << ") failed";
    if (global_core->owner) {
      PCHECK(shm_unlink(global_core->shm_name) == 0)
          << ": shared_mem: shm_unlink(" << global_core->shm_name << ") failed";
    }
    global_core = nullptr;
  }
}

int aos_core_is_init(void) { return global_core != NULL; }
