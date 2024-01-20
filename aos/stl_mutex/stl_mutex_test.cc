#include "aos/stl_mutex/stl_mutex.h"

#include "gtest/gtest.h"

#include "aos/die.h"

namespace aos::testing {

class StlMutexDeathTest : public ::testing::Test {
 protected:
  void SetUp() override { SetDieTestMode(true); }
};

typedef StlMutexDeathTest StlRecursiveMutexDeathTest;

// Tests that locking/unlocking without any blocking works.
TEST(StlMutexTest, Basic) {
  stl_mutex mutex;

  mutex.lock();
  mutex.unlock();

  ASSERT_TRUE(mutex.try_lock());
  ASSERT_FALSE(mutex.try_lock());
  mutex.unlock();

  mutex.lock();
  ASSERT_FALSE(mutex.try_lock());
  mutex.unlock();
}

// Tests that unlocking an unlocked mutex fails.
TEST_F(StlMutexDeathTest, MultipleUnlock) {
  stl_mutex mutex;
  mutex.lock();
  mutex.unlock();
  EXPECT_DEATH(mutex.unlock(), ".*multiple unlock.*");
}

// Tests that locking/unlocking (including recursively) without any blocking
// works.
TEST(StlRecursiveMutexTest, Basic) {
  stl_recursive_mutex mutex;

  mutex.lock();
  mutex.unlock();

  ASSERT_TRUE(mutex.try_lock());
  ASSERT_TRUE(mutex.try_lock());
  mutex.unlock();
  mutex.unlock();

  mutex.lock();
  ASSERT_TRUE(mutex.try_lock());
  mutex.unlock();
  mutex.unlock();
}

// Tests that unlocking an unlocked recursive mutex fails.
TEST_F(StlRecursiveMutexDeathTest, MultipleUnlock) {
  stl_recursive_mutex mutex;
  mutex.lock();
  mutex.unlock();
  EXPECT_DEATH(mutex.unlock(), ".*multiple unlock.*");
}

}  // namespace aos::testing
