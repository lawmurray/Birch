/**
 * @file
 */
#pragma once

#include "libbirch/clone.hpp"
#include "libbirch/Any.hpp"

namespace libbirch {
/**
 * Sets the thread-local variable currentContext on construction, sets it
 * back to its previous value on destruction. This is the recommended way to
 * interact with this global variable, e.g.:
 *
 *     void f() {
 *       SwapContext swapContext(context);
 *       ...
 *     }
 */
class SwapContext {
public:
  /**
   * Constructor.
   *
   * @param context The context to swap in.
   */
  SwapContext(Memo* context) : prevContext(currentContext) {
    currentContext = context;
  }

  /**
   * Constructor.
   *
   * @param o An object; the context to which it belongs will be swapped in.
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
}
