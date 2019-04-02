/**
 * @file
 */
#pragma once

#include "libbirch/memory.hpp"

namespace libbirch {
/**
 * Swap the cross flag on construction, swap it back on destruction.
 */
class SwapCross {
public:
  /**
   * Constructor.
   */
  SwapCross(const bool finish) : prevCrossUnderway(crossUnderway) {
    crossUnderway = finish;
  }

  /**
   * Destructor.
   */
  ~SwapCross() {
    crossUnderway = prevCrossUnderway;
  }

private:
  /**
   * Previous cross flag.
   */
  bool prevCrossUnderway;
};
}
