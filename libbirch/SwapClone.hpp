/**
 * @file
 */
#pragma once

#include "libbirch/memory.hpp"

namespace libbirch {
/**
 * Sets the thread-local clone flag cloneUnderway on construction, swaps it
 * back to its previous value on destruction. This is the recommended way to
 * modify this flag, e.g.:
 *
 *     void f() {
 *       SwapClone swapClone(true);
 *       ...
 *     }
 */
class SwapClone {
public:
  /**
   * Constructor.
   *
   * @param clone Value to which to set the flag.
   */
  SwapClone(const bool clone) : prevCloneUnderway(cloneUnderway) {
    cloneUnderway = clone;
  }

  /**
   * Destructor.
   */
  ~SwapClone() {
    cloneUnderway = prevCloneUnderway;
  }

private:
  /**
   * Previous clone flag.
   */
  bool prevCloneUnderway;
};
}
