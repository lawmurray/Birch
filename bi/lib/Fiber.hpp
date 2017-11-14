/**
 * @file
 */
#pragma once

#include "bi/lib/AllocationMap.hpp"
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
      allocationMap(
          closed ?
              new AllocationMap(*fiberAllocationMap) : nullptr),
      closed(closed) {
    //
  }

  /**
   * Copy constructor.
   */
  Fiber(const Fiber<Type>& o) :
      state(o.state),
      closed(o.closed) {
    if (closed) {
      allocationMap = new AllocationMap(*o.allocationMap);
    } else {
      allocationMap = nullptr;
    }
  }

  /**
   * Move constructor.
   */
  Fiber(Fiber<Type> && o) = default;

  /**
   * Copy assignment.
   */
  Fiber<Type>& operator=(const Fiber<Type>& o) {
    state = o.state;
    closed = o.closed;
    if (closed) {
      allocationMap = new AllocationMap(*o.allocationMap);
    } else {
      allocationMap = nullptr;
    }
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
   * Swap in/out from global variables.
   */
  void swap() {
    if (allocationMap != nullptr) {
      std::swap(allocationMap, fiberAllocationMap);
    }
  }

  /**
   * Fiber state.
   */
  Pointer<FiberState<Type>> state;

  /**
   * Fiber allocation map. If shared with the parent fiber, then nullptr.
   */
  AllocationMap* allocationMap;

  /**
   * Is this a closed fiber?
   */
  bool closed;
};
}
