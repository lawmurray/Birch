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
   * Constructor.
   */
  Fiber(const bool closed = false) :
      heap(closed ? new Heap(*fiberHeap) : nullptr) {
    //
  }

  /**
   * Copy constructor.
   */
  Fiber(const Fiber<Type>& o) {
    if (o.heap) {
      heap = new Heap(*o.heap);
    } else {
      heap = nullptr;
    }
    state = o.state;
  }

  /**
   * Move constructor.
   */
  Fiber(Fiber<Type> && o) : heap(o.heap), state(o.state) {
    o.heap = nullptr;
    o.state = nullptr;
  }

  /**
   * Destructor.
   */
  virtual ~Fiber() {
    if (heap) {
      delete heap;
    }
  }

  /**
   * Copy assignment.
   */
  Fiber<Type>& operator=(const Fiber<Type>& o) {
    if (heap) {
      if (o.heap) {
        *heap = *o.heap;
      } else {
        delete heap;
        heap = nullptr;
      }
    } else if (o.heap) {
      heap = new Heap(*o.heap);
    }
    state = o.state;
    return *this;
  }

  /**
   * Move assignment.
   */
  Fiber<Type>& operator=(Fiber<Type> && o) {
    std::swap(heap, o.heap);
    std::swap(state, o.state);
    return *this;
  }

  /**
   * Run to next yield point.
   *
   * @return Was a value yielded?
   */
  bool query() {
    if (!state.isNull()) {
      swap();
      bool result = state->query();
      swap();
      return result;
    } else {
      return false;
    }
  }

  /**
   * Get the last yield value.
   */
  Type& get() {
    assert(!state.isNull());
    swap();
    Type& result = state->get();
    swap();
    return result;
  }
  const Type& get() const {
    assert(!state.isNull());
    swap();
    const Type& result = state->get();
    swap();
    return result;
  }

  /**
   * Swap in/out the fiber-local heap.
   */
  void swap() {
    if (heap) {
      std::swap(heap, fiberHeap);
    }
  }

  /**
   * Fiber state.
   */
  Pointer<FiberState<Type>> state;

  /**
   * Fiber-local heap, nullptr if shared with the parent fiber.
   */
  Heap* heap;
};
}
