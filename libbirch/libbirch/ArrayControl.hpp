/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
/**
 * Control block for reference counting of Array buffers.
 *
 * @ingroup libbirch
 */
class ArrayControl {
public:
  /**
   * Constructor.
   *
   * @param r Initial reference count.
   */
  ArrayControl(const int r) : r_(r) {
    //
  }

  /**
   * Reference count.
   */
  int numShared_() const {
    return r_.load();
  }

  /**
   * Increment the shared reference count.
   */
  void incShared_() {
    assert(numShared_() > 0);
    r_.increment();
  }

  /**
   * Decrement the shared reference count and return the new value.
   */
  int decShared_() {
    return --r_;
  }

private:
  /**
   * Reference count.
   */
  Atomic<int> r_;
};
}
