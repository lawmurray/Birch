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
  Fiber(FiberState<Type>* state = nullptr, const bool isClosed = false);

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
   *
   * @internal Returns by value to ensure that pointers, from the fiber's
   * world, are mapped to the caller's world.
   */
  const Type get() const;

private:
  /**
   * Mark as dirty.
   *
   * @internal This must be const as it is applied, via a const_cast, to the
   * src fiber when the copy constructor or copy assignment operator are
   * called.
   */
  void dirty() const;

  /**
   * Close.
   */
  void close();

  /**
   * Fiber state.
   */
  std::unique_ptr<FiberState<Type>> state;

  /**
   * Fiber world.
   */
  std::shared_ptr<World> world;

  /**
   * Is this fiber closed?
   */
  bool isClosed;

  /**
   * Is this fiber dirty?
   */
  bool isDirty;
};
}

#include "libbirch/World.hpp"

template<class Type>
bi::Fiber<Type>::Fiber(FiberState<Type>* state, const bool isClosed) :
    state(state),
    world(isClosed ? std::make_shared<World>(fiberWorld) : fiberWorld),
    isClosed(isClosed),
    isDirty(false) {
  assert(state);
}

template<class Type>
bi::Fiber<Type>::Fiber(const Fiber<Type>& o) :
    state(o.state->clone()),
    world(o.isClosed ? std::make_shared<World>(o.world) : o.world),
    isClosed(o.isClosed),
    isDirty(false) {
  o.dirty();
}

template<class Type>
bi::Fiber<Type>& bi::Fiber<Type>::operator=(const Fiber<Type>& o) {
  state.reset(o.state->clone());
  world = o.isClosed ? std::make_shared<World>(o.world) : o.world;
  isClosed = o.isClosed;
  isDirty = false;
  o.dirty();
  return *this;
}

template<class Type>
bool bi::Fiber<Type>::query() {
  bool result = false;
  auto prevWorld = fiberWorld;
  fiberWorld = world;
  result = state->query();
  fiberWorld = prevWorld;
  return result;
}

template<class Type>
const Type bi::Fiber<Type>::get() const {
  return state->get();
}

template<class Type>
void bi::Fiber<Type>::dirty() const {
  const_cast<Fiber<Type>*>(this)->isDirty = true;
}

template<class Type>
void bi::Fiber<Type>::close() {
  world = std::make_shared<World>(world);
  isClosed = true;
  isDirty = false;
}
