/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#include "libbirch/LazyAny.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

libbirch::LazyAny::LazyAny() :
    Counted(),
    context(currentContext),
    forward(nullptr) {
  assert(context);
}

libbirch::LazyAny::LazyAny(const LazyAny& o) :
    Counted(o),
    context(currentContext),
    forward(nullptr) {
  assert(context);
}

libbirch::LazyAny::~LazyAny() {
  LazyAny* forward = this->forward.load(std::memory_order_relaxed);
  if (forward) {
    forward->decShared();
  }
}

libbirch::LazyAny* libbirch::LazyAny::getForward() {
  if (isFrozen()) {
    LazyAny* forward = this->forward.load(std::memory_order_relaxed);
    if (!forward) {
      SwapClone swapClone(true);
      SwapContext swapContext(context.get());
      LazyAny* cloned = this->clone_();
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

libbirch::LazyAny* libbirch::LazyAny::pullForward() {
  if (isFrozen()) {
    LazyAny* forward = this->forward.load(std::memory_order_relaxed);
    if (forward) {
      return forward->pullForward();
    }
  }
  return this;
}

#endif
