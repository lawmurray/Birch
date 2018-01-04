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
  Fiber(Fiber<Type>&& o);

  /**
   * Destructor.
   */
  ~Fiber();

  /**
   * Copy assignment.
   */
  Fiber<Type>& operator=(const Fiber<Type>& o);

  /**
   * Move assignment.
   */
  Fiber<Type>& operator=(Fiber<Type>&& o);

  /**
   * Run to next yield point.
   *
   * @return Was a value yielded?
   */
  bool query();

  /**
   * Get the last yield value.
   *
   * @internal Returns by value to ensure that pointers, from the fiber's
   * world, are mapped to the caller's world.
   */
  const Type get() const;

private:
  /**
   * Fiber state.
   *
   * @internal To avoid confusion, this is managed explicitly by the Fiber
   * class, rather than implicitly by using a SharedPointer. Closed fibers
   * are the interface between worlds, and SharedPointer can cause confusion
   * as to which world a fiber's state should actually belong to.
   */
  FiberState<Type>* state;

  /**
   * World of the fiber. Zero for an open fiber.
   */
  world_t world;
};
}

#include "libbirch/AllocationMap.hpp"
#include "libbirch/Allocation.hpp"

template<class Type>
bi::Fiber<Type>::Fiber(FiberState<Type>* state, const bool closed) :
   state(state),
   world(closed ? ++nworlds : 0) {
  if (world > 0) {
    /* the currently running fiber has exported from its world, which must
     * now become read only, with modifications copy-on-write */
    fiberWorld = ++nworlds;
  }
}

template<class Type>
bi::Fiber<Type>::Fiber(const Fiber<Type>& o) :
    world((o.world > 0) ? ++nworlds : 0) {
  if (o.state) {
    auto prevWorld = fiberWorld;
    if (world > 0) {
      fiberWorld = world;
    }
    state = o.state->clone();
    if (world > 0) {
      assert(world == fiberWorld);  // shouldn't change
      fiberWorld = prevWorld;
    }
    if (o.world > 0) {
      /* the copied fiber has exported from its world, which must
       * now become read only, with modifications copy-on-write */
      const_cast<Fiber<Type>&>(o).world = ++nworlds;
    }
  } else {
    state = nullptr;
  }
}

template<class Type>
bi::Fiber<Type>::Fiber(Fiber<Type>&& o) :
    state(o.state),
    world(o.world) {
  o.state = nullptr;
}

template<class Type>
bi::Fiber<Type>::~Fiber() {
  delete state;
}

template<class Type>
bi::Fiber<Type>& bi::Fiber<Type>::operator=(const Fiber<Type>& o) {
  world = (o.world > 0) ? ++nworlds : 0;

  if (o.state) {
    auto prevWorld = fiberWorld;
    if (world > 0) {
      fiberWorld = world;
    }
    state = o.state->clone();
    if (world > 0) {
      assert(world == fiberWorld);  // shouldn't change
      fiberWorld = prevWorld;
    }
    if (o.world > 0) {
      /* the source fiber has exported from its world, which must
       * now become read only, with modifications copy-on-write */
      const_cast<Fiber<Type>&>(o).world = ++nworlds;
    }
  } else {
    state = nullptr;
  }
  return *this;
}

template<class Type>
bi::Fiber<Type>& bi::Fiber<Type>::operator=(Fiber<Type>&& o) {
  std::swap(world, o.world);
  std::swap(state, o.state);
  return *this;
}

template<class Type>
bool bi::Fiber<Type>::query() {
  bool result = false;
  if (state) {
    auto prevWorld = fiberWorld;
    if (world > 0) {
      fiberWorld = world;
    }
    result = state->query();
    if (world > 0) {
      world = fiberWorld;
      fiberWorld = prevWorld;
    }
    if (world > 0 && !state->yieldIsValue()) {
      /* this fiber, which is yielding, has exported from its world, which must
       * now become read only, with modifications copy-on-write */
      world = ++nworlds;
    }
  }
  return result;
}

template<class Type>
const Type bi::Fiber<Type>::get() const {
  assert(state);
  return state->get();
}
