/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

namespace bi {
/**
 * Auxiliary class for switching the clone flag. This is used on the stack
 * to switch the flag for the lifetime of a temporary variable:
 *
 *     Clone clone;
 */
class Clone {
public:
  /**
   * Constructor.
   */
  Clone() :
      prevClone(fiberClone) {
    fiberClone = true;
  }

  /**
   * Destructor.
   */
  ~Clone() {
    fiberClone = prevClone;
  }

private:
  /**
   * The previous clone flag.
   */
  bool prevClone;
};
}
