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
  SwapContext(Context* context) : prevContext(currentContext) {
    currentContext = context;
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
  Context* prevContext;
};
}
