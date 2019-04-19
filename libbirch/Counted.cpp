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
      frozen.store(libbirch::nthreads);  // release the lock and mark as frozen
      ptr->decShared();
    }
  } else if (expected == libbirch::tid) {
    /* this thread already has the lock, but has rediscovered this object,
     * proceed */
  } else {
    /* another thread is in the process of freezing this object, spin until
     * it has finished */
    while (frozen.load() != libbirch::nthreads) {
      //
    }
  }
}
