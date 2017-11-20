/**
 * @file
 */
#pragma once

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
  Fiber(const bool closed = false);

  /**
   * Copy constructor.
   */
  Fiber(const Fiber<Type>& o);

  /**
   * Move constructor.
   */
  Fiber(Fiber<Type> && o) = default;

  /**
   * Copy assignment.
   */
  Fiber<Type>& operator=(const Fiber<Type>& o);

  /**
   * Move assignment.
   */
  Fiber<Type>& operator=(Fiber<Type> && o) = default;

  /**
   * Run to next yield point.
   *
   * @return Was a value yielded?
   */
  bool query();

  /**
   * Get the last yield value.
   */
  Type& get();
  const Type& get() const;

  /**
   * Swap in/out from global variables.
   */
  void swap();

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

#include "bi/lib/AllocationMap.hpp"

template<class Type>
bi::Fiber<Type>::Fiber(const bool closed) :
    allocationMap(closed ? new (GC) AllocationMap(*fiberAllocationMap) : nullptr),
    closed(closed) {
  //
}

template<class Type>
bi::Fiber<Type>::Fiber(const Fiber<Type>& o) :
    state(o.state),
    closed(o.closed) {
  if (closed) {
    allocationMap = new (GC) AllocationMap(*o.allocationMap);
  } else {
    allocationMap = nullptr;
  }
}

template<class Type>
bi::Fiber<Type>& bi::Fiber<Type>::operator=(const Fiber<Type>& o) {
  if (o.closed) {
    //if (closed) {
    //  *allocationMap = *o.allocationMap;
    //} else {
      allocationMap = new (GC) AllocationMap(*o.allocationMap);
    //}
  } else {
    allocationMap = nullptr;
  }
  state = o.state;
  closed = o.closed;
  return *this;
}

template<class Type>
bool bi::Fiber<Type>::query() {
  if (!state.isNull()) {
    swap();
    bool result = state->query();
    swap();
    return result;
  } else {
    return false;
  }
}

template<class Type>
Type& bi::Fiber<Type>::get() {
  assert(!state.isNull());
  swap();
  auto result = state->get();
  swap();
  return result;
}

template<class Type>
const Type& bi::Fiber<Type>::get() const {
  assert(!state.isNull());
  swap();
  auto result = state->get();
  swap();
  return result;
}

template<class Type>
void bi::Fiber<Type>::swap() {
  if (allocationMap != nullptr) {
    std::swap(allocationMap, fiberAllocationMap);
  }
}
