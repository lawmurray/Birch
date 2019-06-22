/**
 * @file
 */
#include "libbirch/Counted.hpp"

void libbirch::Counted::freeze() {
  if (!frozen.load()) {
    auto value = (numShared() <= 1u && numWeak() - numMemo() == 1u) ? 2u : 1u;
    if (!frozen.exchange(value)) {
      /* this thread has obtained the lock, go ahead and freeze */
      auto ptr = lock();
      if (ptr) {
        doFreeze_();
        ptr->decShared();
      }
    }
  }
}
