/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/memory.hpp"

namespace bi {
/**
 * Swap the clone flag on construction, swap it back on destruction.
 */
class SwapClone {
public:
  /**
   * Constructor.
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
