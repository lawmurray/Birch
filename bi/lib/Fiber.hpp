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
 * @tparam Type Yield type.
 */
template<class Type>
class Fiber {
public:
  /**
   * Default constructor.
   */
  Fiber() : heap(fiberHeap), own(false) {
    //
  }

  /**
   * Copy constructor. Fibers are copy-by-value.
   */
  Fiber(const Fiber<Type>& o) : state(o.state) {
    heap = new Heap(*o.heap);
    own = true;
  }

  /**
   * Move constructor.
   */
  Fiber(Fiber<Type>&& o) = default;

  /**
   * Destructor.
   */
  virtual ~Fiber() {
    if (own) {
      delete heap;
    }
  }

  /**
   * Copy assignment.
   */
  Fiber<Type>& operator=(const Fiber<Type>& o) {
    state = o.state;
    if (own) {
      delete heap;
    }
    heap = new Heap(*o.heap);
    own = true;
    return *this;
  }

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
    fiberHeap = heap;
    bool result = state->query();
    fiberHeap = callerHeap;
    return result;
  }

  /**
   * Get the last yield value.
   */
  Type& get() {
    Heap* callerHeap = fiberHeap;
    fiberHeap = heap;
    Type& result = state->get();
    fiberHeap = callerHeap;
    return result;
  }
  const Type& get() const {
    Heap* callerHeap = fiberHeap;
    fiberHeap = heap;
    const Type& result = state->get();
    fiberHeap = callerHeap;
    return result;
  }

  /**
   * Fiber state.
   */
  Pointer<FiberState<Type>> state;

  /**
   * Fiber-local heap, nullptr if shared with the parent fiber.
   */
  Heap* heap;

  /**
   * Does the fiber own this heap?
   */
  bool own;
};
}
