/**
 * @file
 */
#include "libbirch/Lock.hpp"

bi::Lock::Lock() : lock({0,0}) {
  //
}

void bi::Lock::share() {
  /* spin until exclusive lock is released and shared count updated */
  lock_type expected = lock, desired;
  do {
    expected.keepCount = 0;
    desired = {expected.shareCount + 1, 0};
  } while (lock.compare_exchange_weak(expected, desired));
}

void bi::Lock::unshare() {
  lock_type expected = lock, desired;
  do {
    desired = {expected.shareCount - 1, expected.keepCount};
  } while (lock.compare_exchange_weak(expected, desired));
}

void bi::Lock::keep() {
  /* spin until exclusive lock is obtained */
  lock_type expected = lock, desired;
  do {
    expected.keepCount = 0;
    desired = {expected.shareCount, 1};
  } while (lock.compare_exchange_weak(expected, desired));
}

void bi::Lock::unkeep() {
  lock = {0, 0};
}
