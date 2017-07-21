/**
 * @file
 */
#pragma once

#include "bi/lib/Heap.hpp"

namespace bi {
/**
 * Relocatable fiber.
 *
 * @ingroup library
 *
 * @tparam Type Return type.
 */
template<class Type>
class Fiber : public Heap {
public:
  /**
   * Constructor.
   */
  Fiber() : yieldTo(this), state(0) {
    //
  }

  /**
   * Run to next yield point.
   */
  Type operator()() {
    std::swap(yieldTo, currentFiber);
    Type result = run();
    std::swap(yieldTo, currentFiber);
    return result;
  }

protected:
  /**
   * Run to next yield point.
   */
  virtual Type run() = 0;

  /**
   * Fiber to which to yield at next yield point.
   */
  Heap* yieldTo;

  /**
   * State.
   */
  int state;
};
}
