/**
 * @file
 */
#pragma once

#include "libbirch/clone.hpp"
#include "libbirch/Any.hpp"

namespace libbirch {
#pragma omp declare target
/**
 * Swap the context on construction, swap it back on destruction.
 */
class SwapContext {
public:
  /**
   * Constructor.
   */
  SwapContext(Memo* context) : prevContext(currentContext) {
    currentContext = context;
  }

  /**
   * Constructor.
   */
  SwapContext(Any* o) : SwapContext(o->getContext()) {
    //
  }

  /**
   * Destructor.
   */
  ~SwapContext() {
    currentContext = prevContext;
  }

private:
  /**
   * Previous context.
   */
  Memo* prevContext;
};
#pragma omp end declare target
}
