/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/memory.hpp"

bi::Any::Any() :
    context(currentContext),
    forward(nullptr),
    freezeCount(0) {
  //
}

bi::Any::Any(const Any& o) :
    Counted(o),
    context(currentContext),
    forward(nullptr),
    freezeCount(0) {
  //
}

bi::Any::~Any() {
  Any* forward = this->forward;  // load from atomic just once
  if (forward) {
    forward->decShared();
  }
}

bi::Any* bi::Any::getForward() {
  if (isFrozen()) {
    Any* forward = this->forward;  // load from atomic just once
    if (!forward) {
      SwapClone swapClone(true);
      SwapContext swapContext(context.get());
      Any* cloned = this->clone();
      cloned->incShared();
      if (this->forward.compare_exchange_strong(forward, cloned)) {
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

bi::Any* bi::Any::pullForward() {
  if (isFrozen()) {
    Any* forward = this->forward;  // load from atomic just once
    if (forward) {
      return forward->pullForward();
    }
  }
  return this;
}

bi::Memo* bi::Any::getContext() {
  return context.get();
}

bool bi::Any::isFrozen() const {
  unsigned n;
  do {
    n = freezeCount.load();
  } while (n > 0u && n < nthreads + 1u);
  return freezeCount > 0u;
}

void bi::Any::freeze() {
  unsigned expected = 0u;
  unsigned desired = tid + 1;
  if (freezeCount.compare_exchange_strong(expected, desired)) {
    /* this thread freezes the object */
    doFreeze();
    freezeCount = nthreads + 1u;
  } else if (expected < nthreads + 1u) {
    /* this thread is currently freezing the object, nothing to do */
  } else if (expected < nthreads + 1u) {
    /* another thread is currently freezing the object, join it, if for no
     * other reason than to avoid a potential deadlock situation */
    doFreeze();
    freezeCount = nthreads + 1u;
  } else {
    /* already frozen */
    assert(expected == nthreads + 1u);
  }
}

void bi::Any::doFreeze() {
  //
}
