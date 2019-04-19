/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#include "libbirch/LazyAny.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

libbirch::LazyAny* libbirch::LazyAny::getForward() {
  assert(isFrozen());

  auto forward1 = forward.load(std::memory_order_relaxed);
  if (!forward1) {
    SwapClone swapClone(true);
    SwapContext swapContext(this);
    auto forward2 = this->clone_();
    forward2->incShared();
    if (this->forward.compare_exchange_strong(forward1, forward2,
        std::memory_order_relaxed)) {
      return forward2;
    } else {
      /* beaten by another thread */
      forward2->decShared();
    }
  }
  return forward1->getForward();
}

#endif
