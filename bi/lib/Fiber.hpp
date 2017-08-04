/**
 * @file
 */
#pragma once

#include "bi/lib/Heap.hpp"
#include "bi/lib/FiberState.hpp"

namespace bi {
/**
 * Fiber.
 *
 * @ingroup library
 *
 * @tparam Type Return type.
 */
template<class Type>
class Fiber: public Heap {
public:
  /**
   * Default constructor.
   */
  Fiber() {
    //
  }

  /**
   * Copy constructor. Fibers are copy-by-value.
   */
  Fiber(const Fiber<Type>& o) = default;

  /**
   * Move constructor.
   */
  Fiber(Fiber<Type>&& o) = default;

  /**
   * Copy assignment.
   */
  Fiber<Type>& operator=(const Fiber<Type>& o) = default;

  /**
   * Move assignment.
   */
  Fiber<Type>& operator=(Fiber<Type> && o) = default;

  /**
   * Run to next yield point.
   *
   * @return Was a value yielded?
   */
  bool query() {
    Heap* yieldTo = currentFiber;
    currentFiber = this;
    bool result = state->query();
    currentFiber = yieldTo;
    return result;
  }

  /**
   * Get the last yield value.
   */
  Type& get() {
    Heap* yieldTo = currentFiber;
    currentFiber = this;
    Type& result = state->get();
    currentFiber = yieldTo;
    return result;
  }
  const Type& get() const {
    Heap* yieldTo = currentFiber;
    currentFiber = this;
    const Type& result = state->get();
    currentFiber = yieldTo;
    return result;
  }

  /**
   * Fiber state.
   */
  Pointer<FiberState<Type>> state;
};
}
