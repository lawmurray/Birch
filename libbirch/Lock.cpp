/**
 * @file
 */
#include "libbirch/Lock.hpp"

bi::Lock::Lock() {
  lock.joint = {0u, 0u};
}

void bi::Lock::share() {
  /* spin until exclusive lock is released and shared count updated */
  joint_lock_type expected = lock.joint, desired;
  do {
    expected.keepCount = 0;
    desired = {expected.shareCount + 1, 0};
  } while (!lock.joint.compare_exchange_weak(expected, desired));
}

void bi::Lock::unshare() {
  --lock.split.shareCount;
}

void bi::Lock::keep() {
  /* spin until exclusive lock obtained */
  unsigned expected;
  do {
    expected = 0;
  } while (!lock.split.keepCount.compare_exchange_weak(expected, 1));

  /* spin until all threads with shared locks release */
  while (lock.split.shareCount > 0);
}

void bi::Lock::unkeep() {
  lock.split.keepCount = 0;
}
