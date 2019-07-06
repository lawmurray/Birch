/**
 * @file
 */
#pragma once

#include "libbirch/memory.hpp"

namespace libbirch {
/**
 * Sets the thread-local flag finishUnderway on construction, swaps it
 * back to its previous value on destruction.
 */
class SwapFinish {
public:
  /**
   * Constructor.
   *
   * @param value Value to which to set the flag.
   */
  SwapFinish(const bool value) : previous(finishUnderway) {
    finishUnderway = value;
  }

  /**
   * Destructor.
   */
  ~SwapFinish() {
    finishUnderway = previous;
  }

private:
  /**
   * Previous value.
   */
  bool previous;
};
}
