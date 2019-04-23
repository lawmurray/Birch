/**
 * @file
 */
#include "libbirch/Counted.hpp"

void libbirch::Counted::freeze() {
  int expected = -1;
  int desired = libbirch::tid;
  if (frozen.compare_exchange_strong(expected, desired)) {
    /* this thread has obtained the lock, go ahead and freeze */
    auto ptr = lock();
    if (ptr) {
      doFreeze_();
      ptr->decShared();
    }

    /* release the lock and mark as frozen, and possibly uniquely
     * reachable */
    if (numShared() <= 1u && numWeak() - numMemo() == 1u) {
      frozen.store(libbirch::nthreads + 1u);
    } else {
      frozen.store(libbirch::nthreads);
    }
  } else if (expected == (int)libbirch::tid) {
    /* this thread already has the lock, but has rediscovered this object,
     * proceed */
  } else {
    /* another thread is in the process of freezing this object, spin until
     * it has finished */
    while (frozen.load() < (int)libbirch::nthreads) {
      //
    }
  }
}

void libbirch::Counted::notUniquelyReachable() {
  if (isFrozen() && (numShared() > 1u || numWeak() - numMemo() != 1u)) {
    int expected = libbirch::nthreads + 1u;
    int desired = libbirch::nthreads;
    frozen.compare_exchange_strong(expected, desired);
  }
}
