/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/clone.hpp"

namespace bi {
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
