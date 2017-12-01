/**
 * @file
 */
#pragma once

#include "libbirch/FiberState.hpp"
#include "libbirch/FiberWorld.hpp"

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
   * Fiber state.
   */
  Pointer<FiberState<Type>> state;

  /**
   * World of the fiber.
   */
  FiberWorld* world;

  /**
   * Is this a closed fiber?
   */
  const bool closed;
};
}

#include "libbirch/AllocationMap.hpp"

template<class Type>
bi::Fiber<Type>::Fiber(FiberState<Type>* state, const bool closed) :
    state(state),
    closed(closed) {
  if (closed) {
    /* the currently running fiber has exported from its world, which must
     * now become read only, with modifications copy-on-write */
    world = new (GC_MALLOC(sizeof(FiberWorld))) FiberWorld(fiberWorld);
    fiberWorld = new (GC_MALLOC(sizeof(FiberWorld))) FiberWorld(fiberWorld);
  } else {
    world = fiberWorld;
  }
}

template<class Type>
bi::Fiber<Type>::Fiber(const Fiber<Type>& o) :
    state(o.state),
    closed(o.closed) {
  if (closed) {
    /* the copied fiber has exported from its world, which must
     * now become read only, with modifications copy-on-write */
    world = new (GC_MALLOC(sizeof(FiberWorld))) FiberWorld(o.world);
    const_cast<Fiber<Type>&>(o).world = new (GC_MALLOC(sizeof(FiberWorld))) FiberWorld(o.world);
  } else {
    world = o.world;
  }
}

template<class Type>
bi::Fiber<Type>& bi::Fiber<Type>::operator=(const Fiber<Type>& o) {
  state = o.state;
  closed = o.closed;
  if (closed) {
    /* the copied fiber has exported from its world, which must
     * now become read only, with modifications copy-on-write */
    world = new (GC_MALLOC(sizeof(FiberWorld))) FiberWorld(o.world);
    const_cast<Fiber<Type>&>(o).world = new (GC_MALLOC(sizeof(FiberWorld))) FiberWorld(o.world);
  } else {
    world = o.world;
  }
  return *this;
}

template<class Type>
bool bi::Fiber<Type>::query() {
  if (!state.isNull()) {
    auto callerWorld = fiberWorld;
    if (closed) {
      fiberWorld = world;
    }
    bool result = state->query();
    if (closed) {
      world = fiberWorld;
      fiberWorld = callerWorld;
    }
    return result;
  } else {
    return false;
  }
}

template<class Type>
Type& bi::Fiber<Type>::get() {
  assert(!state.isNull());

  if (closed && !state->yieldIsValue()) {
    /* this fiber, which is yielding, has exported from its world, which must
     * now become read only, with modifications copy-on-write */
    world = new FiberWorld(world);
  }
  return state->get();
}

template<class Type>
const Type& bi::Fiber<Type>::get() const {
  assert(!state.isNull());

  if (closed && !state->yieldIsValue()) {
    /* this fiber, which is yielding, has exported from its world, which must
     * now become read only, with modifications copy-on-write */
    world = new FiberWorld(world);
  }
  return state->get();
}
