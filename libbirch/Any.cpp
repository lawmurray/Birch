/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/memory.hpp"

bi::Any::Any() :
    context(currentContext),
	  frozen(false) {
  //
}

bi::Any::Any(const Any& o) :
    Counted(o),
    context(currentContext),
	  frozen(false) {
  //
}

bi::Any::~Any() {
  //
}

bool bi::Any::isFrozen() const {
  return frozen;
}

void bi::Any::freeze() {
  frozen = true;
}

bi::Any* bi::Any::getForward() {
  if (frozen) {
    if (!forward) {
      ///@todo Race condition
      SwapClone swapClone(true);
      SwapContext swapContext(context.get());
      forward = this->clone();
    }
    return forward.get();
  } else {
    return this;
  }
}

bi::Memo* bi::Any::getContext() {
  return context.get();
}
