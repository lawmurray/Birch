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
   * Constructor.
   *
   * @param state State of the fiber.
   */
  Fiber(const Pointer<FiberState<Type>>& state = nullptr) :
      state(state) {
    //
  }

  /**
   * Copy constructor. Fibers are copy-by-value.
   */
  Fiber(const Fiber& o) :
      Heap(o),
      state(o.state->clone()) {
    //
  }

  /**
   * Move constructor.
   */
  Fiber(Fiber&& o) :
      Heap(o),
      state(o.state) {
    o.state = nullptr;
  }

  /**
   * Copy assignment.
   */
  Fiber<Type>& operator=(const Fiber<Type>& o) {
    Heap::operator=(o);
    state = o.state->clone();
    return *this;
  }

  /**
   * Move assignment.
   */
  Fiber<Type>& operator=(Fiber<Type> && o) {
    Heap::operator=(o);
    std::swap(state, o.state);
    return *this;
  }

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
    return state->get();
  }
  const Type& get() const {
    return state->get();
  }

protected:
  /**
   * Fiber state.
   */
  Pointer<FiberState<Type>> state;
};
}
