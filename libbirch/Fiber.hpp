/**
 * @file
 */
#pragma once

#include "libbirch/FiberState.hpp"

namespace bi {
/**
 * Fiber.
 *
 * @ingroup libbirch
 *
 * @tparam Type Yield type.
 */
template<class Type>
class Fiber {
public:
  /**
   * Constructor.
   */
  Fiber(FiberState<Type>* state = nullptr, const bool closed = false);

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
   * Fiber allocation map.
   */
  AllocationMap* allocationMap;

  /**
   * Fiber state.
   */
  Pointer<FiberState<Type>> state;

  /**
   * Generation of the fiber.
   */
  size_t gen;

  /**
   * Is this a closed fiber?
   */
  bool closed;
};
}

#include "libbirch/AllocationMap.hpp"

template<class Type>
bi::Fiber<Type>::Fiber(FiberState<Type>* state, const bool closed) :
    allocationMap(
        closed ?
            new (GC) AllocationMap(*fiberAllocationMap) : fiberAllocationMap),
    state(state, fiberGen),
    gen(closed ? ++fiberGen : fiberGen),
    closed(closed) {
  //
}

template<class Type>
bi::Fiber<Type>::Fiber(const Fiber<Type>& o) :
    allocationMap(
        o.closed ?
            new (GC) AllocationMap(*o.allocationMap) : o.allocationMap),
    state(o.state),
    gen(o.closed ? ++const_cast<Fiber<Type>&>(o).gen : o.gen),
    closed(o.closed) {
  //
}

template<class Type>
bi::Fiber<Type>& bi::Fiber<Type>::operator=(const Fiber<Type>& o) {
  allocationMap =
      o.closed ? new (GC) AllocationMap(*o.allocationMap) : o.allocationMap;
  state = o.state;
  gen = o.closed ? ++const_cast<Fiber<Type>&>(o).gen : o.gen;
  closed = o.closed;
  return *this;
}

template<class Type>
bool bi::Fiber<Type>::query() {
  if (!state.isNull()) {
    auto callerAllocationMap = fiberAllocationMap;
    auto callerGen = fiberGen;

    if (closed) {
      fiberGen = gen;
      fiberAllocationMap = allocationMap;
    }

    bool result = state->query();

    if (closed) {
      allocationMap = fiberAllocationMap;
      gen = fiberGen;

      fiberAllocationMap = callerAllocationMap;
      fiberGen = callerGen;
    }

    return result;
  } else {
    return false;
  }
}

template<class Type>
Type& bi::Fiber<Type>::get() {
  assert(!state.isNull());

  ///@todo Yield value may not have been mapped through fiber's allocation
  ///      map, and even if it has, its member attributes may not have been.
  ///      Does the allocation map need to be global?
  ///@todo When caller copies out yield value the first time, entry is put in
  ///      allocation map... this needs to be removed for subsequent calls or
  ///      the value is never copied again, the first copy is just reused.
  ///      Similar applies to its member attributes.
  if (closed && !state->yieldIsValue()) {
    /* update the generation of the fiber retrieving the result to make sure
     * that it will not modify it without copying it */
    fiberGen = std::max(fiberGen, gen + 1);
  }
  return state->get();
}

template<class Type>
const Type& bi::Fiber<Type>::get() const {
  assert(!state.isNull());

  if (closed && !state->yieldIsValue()) {
    /* update the generation of the fiber retrieving the result to make sure
     * that it will not modify it without copying it */
    fiberGen = std::max(fiberGen, gen + 1);
  }
  return state->get();
}
