#ifndef AOS_MUTEX_H_
#define AOS_MUTEX_H_

#include "glog/logging.h"

#include "aos/ipc_lib/aos_sync.h"
#include "aos/macros.h"
#include "aos/type_traits/type_traits.h"

namespace aos {

class Condition;

// An abstraction of a mutex that is easy to implement for environments other
// than Linux too.
// If there are multiple threads or processes contending for the mutex,
// higher priority ones will succeed in locking first,
// and tasks of equal priorities have the same chance of getting the lock.
// To deal with priority inversion, the linux implementation does priority
// inheritance.
// Before destroying a mutex, it is important to make sure it isn't locked.
// Otherwise, the destructor will LOG(FATAL).
class Mutex {
 public:
  // States that signify the result of TryLock.
  enum class State {
    // The mutex was acquired successfully.
    kLocked,
    // TryLock tried to grab the mutex and failed.
    kLockFailed,
    // The previous owner of the mutex died.
    kOwnerDied,
  };

  // Creates an unlocked mutex.
  Mutex() : impl_() {
    static_assert(shm_ok<Mutex>::value,
                  "Mutex is not safe for use in shared memory.");
  }
  // Verifies that it isn't locked.
  //
  // This is important because freeing a locked mutex means there is freed
  // memory in the middle of the robust list, which breaks things horribly.
  ~Mutex() = default;

  // Locks the mutex. If it fails, it calls LOG(FATAL).
  // Returns true if the previous owner died instead of unlocking nicely.
  bool Lock() __attribute__((warn_unused_result));
  // Unlocks the mutex. Fails like Lock.
  // Multiple unlocking is undefined.
  void Unlock();
  // Locks the mutex unless it is already locked.
  // Returns the new state of the mutex.
  // Doesn't wait for the mutex to be unlocked if it is locked.
  State TryLock() __attribute__((warn_unused_result));

  // Returns true iff the current task has this mutex locked.
  // This is mainly for IPCRecursiveMutexLocker to use.
  bool OwnedBySelf() const;

 private:
  aos_mutex impl_;

  friend class Condition;  // for access to impl_
};

// A class that locks a Mutex when constructed and unlocks it when destructed.
// Designed to be used as a local variable so that
// the mutex will be unlocked when the scope is exited.
// This one immediately Dies if the previous owner died. This makes it a good
// choice for mutexes that are only used within a single process, but NOT for
// mutexes shared by multiple processes. For those, use IPCMutexLocker.
class MutexLocker {
 public:
  explicit MutexLocker(Mutex *mutex) : mutex_(mutex) {
    if (__builtin_expect(mutex_->Lock(), false)) {
      LOG(FATAL) << "previous owner of mutex " << this
                 << " died but it shouldn't be able to";
    }
  }
  ~MutexLocker() { mutex_->Unlock(); }

 private:
  Mutex *const mutex_;

  DISALLOW_COPY_AND_ASSIGN(MutexLocker);
};

// A version of MutexLocker which reports the previous owner dying instead of
// immediately LOG(FATAL)ing.
class IPCMutexLocker {
 public:
  explicit IPCMutexLocker(Mutex *mutex)
      : mutex_(mutex), owner_died_(mutex_->Lock()) {}
  ~IPCMutexLocker() {
    if (__builtin_expect(!owner_died_checked_, false)) {
      LOG(FATAL) << "nobody checked if the previous owner of mutex " << this
                 << " died";
    }
    mutex_->Unlock();
  }

  // Whether or not the previous owner died. If this is not called at least
  // once, the destructor will LOG(FATAL)
  __attribute__((warn_unused_result)) bool owner_died() {
    owner_died_checked_ = true;
    return __builtin_expect(owner_died_, false);
  }

 private:
  Mutex *const mutex_;
  const bool owner_died_;
  bool owner_died_checked_ = false;

  DISALLOW_COPY_AND_ASSIGN(IPCMutexLocker);
};

// A version of IPCMutexLocker which only locks (and unlocks) the mutex if the
// current task does not already hold it.
class IPCRecursiveMutexLocker {
 public:
  explicit IPCRecursiveMutexLocker(Mutex *mutex)
      : mutex_(mutex),
        locked_(!mutex_->OwnedBySelf()),
        owner_died_(locked_ ? mutex_->Lock() : false) {}
  ~IPCRecursiveMutexLocker() {
    if (__builtin_expect(!owner_died_checked_, false)) {
      LOG(FATAL) << "nobody checked if the previous owner of mutex " << this
                 << " died";
    }
    if (locked_) mutex_->Unlock();
  }

  // Whether or not the previous owner died. If this is not called at least
  // once, the destructor will LOG(FATAL)
  __attribute__((warn_unused_result)) bool owner_died() {
    owner_died_checked_ = true;
    return __builtin_expect(owner_died_, false);
  }

 private:
  Mutex *const mutex_;
  const bool locked_, owner_died_;
  bool owner_died_checked_ = false;

  DISALLOW_COPY_AND_ASSIGN(IPCRecursiveMutexLocker);
};

}  // namespace aos

#endif  // AOS_MUTEX_H_
