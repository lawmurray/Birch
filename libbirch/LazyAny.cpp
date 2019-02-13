/**
 * @file
 */
#if USE_LAZY_DEEP_CLONE
#include "libbirch/LazyAny.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

bi::LazyAny::LazyAny() :
    Counted(),
    context(currentContext),
    forward(nullptr) {
  assert(context);
}

bi::LazyAny::LazyAny(const LazyAny& o) :
    Counted(o),
    context(currentContext),
    forward(nullptr) {
  assert(context);
}

bi::LazyAny::~LazyAny() {
  LazyAny* forward = this->forward.load(std::memory_order_relaxed);
  if (forward) {
    forward->decShared();
  }
}

bi::LazyAny* bi::LazyAny::getForward() {
  if (isFrozen()) {
    LazyAny* forward = this->forward.load(std::memory_order_relaxed);
    if (!forward) {
      SwapClone swapClone(true);
      SwapContext swapContext(context.get());
      LazyAny* cloned = this->clone();
      cloned->incShared();
      if (this->forward.compare_exchange_strong(forward, cloned,
          std::memory_order_relaxed)) {
        return cloned;
      } else {
        /* beaten by another thread */
        cloned->decShared();
      }
    }
    return forward->getForward();
  } else {
    return this;
  }
}

bi::LazyAny* bi::LazyAny::pullForward() {
  if (isFrozen()) {
    LazyAny* forward = this->forward.load(std::memory_order_relaxed);
    if (forward) {
      return forward->pullForward();
    }
  }
  return this;
}

#endif
