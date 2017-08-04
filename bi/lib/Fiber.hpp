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
class Fiber {
public:
  /**
   * Default constructor.
   */
  Fiber() {
    if (fiberHeap) {
      heap = *fiberHeap;
    }
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
    Heap* callerHeap = fiberHeap;
    fiberHeap = &heap;
    bool result = state->query();
    fiberHeap = callerHeap;
    return result;
  }

  /**
   * Get the last yield value.
   */
  Type& get() {
    Heap* callerHeap = fiberHeap;
    fiberHeap = &heap;
    Type& result = state->get();
    fiberHeap = callerHeap;
    return result;
  }
  const Type& get() const {
    Heap* callerHeap = fiberHeap;
    fiberHeap = &heap;
    const Type& result = state->get();
    fiberHeap = callerHeap;
    return result;
  }

  /**
   * Fiber state.
   */
  Pointer<FiberState<Type>> state;

  /**
   * Fiber heap.
   */
  Heap heap;
};
}
