/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/memory.hpp"

bi::Any::Any() :
    Counted(),
    context(currentContext),
    forward(nullptr) {
  //
}

bi::Any::Any(const Any& o) :
    Counted(o),
    context(currentContext),
    forward(nullptr) {
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
