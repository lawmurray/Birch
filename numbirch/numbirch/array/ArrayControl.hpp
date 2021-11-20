/**
 * @file
 */
#pragma once

#include "numbirch/array/external.hpp"
#include "numbirch/array/Atomic.hpp"

namespace numbirch {
/**
 * @internal
 * 
 * Control block for reference counting of Array buffers.
 * 
 * @ingroup array
 */
class ArrayControl {
public:
  /**
   * Constructor.
   *
   * @param r Initial reference count.
   */
  ArrayControl(const int r) : r(r) {
    //
  }

  /**
   * Reference count.
   */
  int numShared() const {
    return r.load();
  }

  /**
   * Increment the shared reference count.
   */
  void incShared() {
    assert(numShared() > 0);
    r.increment();
  }

  /**
   * Decrement the shared reference count and return the new value.
   */
  int decShared() {
    return --r;
  }

private:
  /**
   * Reference count.
   */
  Atomic<int> r;
};
}
