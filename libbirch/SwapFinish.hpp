/**
 * @file
 */
#pragma once

#include "libbirch/memory.hpp"

namespace libbirch {
/**
 * Swap the finish flag on construction, swap it back on destruction.
 */
class SwapFinish {
public:
  /**
   * Constructor.
   */
  SwapFinish(const bool finish) : prevFinishUnderway(finishUnderway) {
    finishUnderway = finish;
  }

  /**
   * Destructor.
   */
  ~SwapFinish() {
    finishUnderway = prevFinishUnderway;
  }

private:
  /**
   * Previous finish flag.
   */
  bool prevFinishUnderway;
};
}
