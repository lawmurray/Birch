/**
 * @file
 */
#pragma once

#include "bi/lib/Heap.hpp"
#include "bi/lib/Coroutine.hpp"

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
   *
   * @param coroutine Coroutine associated with the fiber.
   */
  Fiber(Pointer<Coroutine<Type>> coroutine) :
      coroutine(coroutine) {
    //
  }

  /**
   * Copy constructor. Fibers are copy-by-value.
   */
  Fiber(const Fiber& o) :
      coroutine(o.coroutine->clone()) {
    //
  }

  /**
   * Run to next yield point.
   */
  Type operator()() {
    Heap* yieldTo = currentFiber;
    currentFiber = this;
    Type result = (*coroutine)();
    currentFiber = yieldTo;
    return result;
  }

protected:
  /**
   * Coroutine associated with this fiber.
   */
  Pointer<Coroutine<Type>> coroutine;
};
}
