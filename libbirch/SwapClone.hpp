/**
 * @file
 */
#pragma once

#include "libbirch/memory.hpp"

namespace libbirch {
/**
 * Sets the thread-local flag cloneUnderway on construction, swaps it
 * back to its previous value on destruction.
 */
class SwapClone {
public:
  /**
   * Constructor.
   *
   * @param value Value to which to set the flag.
   */
  SwapClone(const bool value) : previous(cloneUnderway) {
    cloneUnderway = value;
  }

  /**
   * Destructor.
   */
  ~SwapClone() {
    cloneUnderway = previous;
  }

private:
  /**
   * Previous value.
   */
  bool previous;
};
}
