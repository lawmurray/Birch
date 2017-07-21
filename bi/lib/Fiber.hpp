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
class Fiber: public Heap {
public:
  /**
   * Constructor.
   */
  Fiber() :
      state(0) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~Fiber() {
    //
  }

  /**
   * Run to next yield point.
   */
  Type operator()() {
    Heap* yieldTo = currentFiber;
    currentFiber = this;
    Type result = run();
    currentFiber = yieldTo;
    return result;
  }

protected:
  /**
   * Run to next yield point.
   */
  virtual Type run() = 0;

  /**
   * State.
   */
  int state;
};
}
