/**
 * @file
 */
#pragma once

#include "libbirch/memory.hpp"

namespace libbirch {
/**
 * Sets the thread-local flag freezeUnderway on construction, swaps it
 * back to its previous value on destruction.
 */
class SwapFreeze {
public:
  /**
   * Constructor.
   *
   * @param value Value to which to set the flag.
   */
  SwapFreeze(const bool value) : previous(freezeUnderway) {
    freezeUnderway = value;
  }

  /**
   * Destructor.
   */
  ~SwapFreeze() {
    freezeUnderway = previous;
  }

private:
  /**
   * Previous value.
   */
  bool previous;
};
}
