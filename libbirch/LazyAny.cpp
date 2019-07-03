/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#include "libbirch/LazyAny.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

libbirch::LazyAny* libbirch::LazyAny::getForward() {
  assert(isFrozen());

  if (!forward) {
    mutex.keep();
    if (!forward) {
      SwapClone swapClone(true);
      SwapContext swapContext(getContext());
      forward = this->clone_();
    }
    mutex.unkeep();
  }
  if (forward->isFrozen()) {
    return forward->getForward();
  } else {
    return forward.get();
  }
}

libbirch::LazyAny* libbirch::LazyAny::pullForward() {
  assert(isFrozen());

  if (forward) {
    if (forward->isFrozen()) {
      return forward->pullForward();
    } else {
      return forward.get();
    }
  } else {
    return this;
  }
}

#endif
