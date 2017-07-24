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
  Fiber(const Pointer<Coroutine<Type>>& coroutine = nullptr) :
      coroutine(coroutine) {
    //
  }

  /**
   * Copy constructor. Fibers are copy-by-value.
   */
  Fiber(const Fiber& o) :
      Heap(o),
      coroutine(o.coroutine->clone()) {
    //
  }

  /**
   * Move constructor.
   */
  Fiber(Fiber&& o) :
      Heap(o),
      coroutine(o.coroutine) {
    o.coroutine = nullptr;
  }

  /**
   * Copy assignment.
   */
  Fiber<Type>& operator=(const Fiber<Type>& o) {
    Heap::operator=(o);
    coroutine = o.coroutine->clone();
    return *this;
  }

  /**
   * Move assignment.
   */
  Fiber<Type>& operator=(Fiber<Type> && o) {
    Heap::operator=(o);
    std::swap(coroutine, o.coroutine);
    return *this;
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
