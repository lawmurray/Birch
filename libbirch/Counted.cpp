/**
 * @file
 */
#include "libbirch/Counted.hpp"

void libbirch::Counted::freeze() {
  auto state = frozen.load();
  if (state > nthreads) {
    // already frozen
  } else if (state == tid + 1u) {
    /* this thread in the process of freezing this object but has
     * rediscovered it; nothing to do */
  } else if (state == 0u) {
    /* not yet frozen, record that this thread is freezing */
    state = frozen.exchange(tid + 1);
    if (state > nthreads) {
      /* another thread wrote final state in the meantime; put it back, and
       * we know the object is now frozen too */
      frozen.store(state);
    } else {
      /* freeze; possible that state is some other thread id here, meaning
       * another thread is in the process of freezing too, but this is
       * harmless */
      auto ptr = lock();
      if (ptr) {
        doFreeze_();
        decShared();
      }
      frozen.store(nthreads + ((numShared() <= 1u && numWeak() - numMemo() == 1u) ? 2u : 1u));
    }
  }
}
