/**
 * @file
 */
#include "libbirch/Counted.hpp"

void libbirch::Counted::freeze() {
  auto state = frozen.load();
  if (state > libbirch::nthreads || state == libbirch::tid + 1u) {
    /* object is already frozen, or this thread is in the process of freezing
     * it but has rediscovered it; nothing to do */
    return;
  } else {
    if (state == 0u) {
      /* not yet frozen, record that this thread is freezing */
      state = frozen.exchange(libbirch::tid + 1u);
      if (state > libbirch::nthreads) {
        /* another thread wrote final state in the meantime; put it back, and
         * now we know that the object is frozen too */
        frozen.store(state);
        return;
      }
    } else if (state <= libbirch::nthreads) {
      /* another thread is in the process of freezing, but this is harmless,
       * continue in order to parallelize the freeze and also ensure that we
       * don't return until all objects reachable from this are frozen */
    }
    if (numShared() > 0u) {
      doFreeze_();
    }

    /* store nthreads + 1 for frozen, nthreads + 2 for frozen and uniquely
     * reachable */
    frozen.store(libbirch::nthreads +
        ((numShared() <= 1u && numWeak() - numMemo() == 1u) ? 2u : 1u));
  }
}
