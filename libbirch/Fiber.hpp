/**
 * @file
 */
#pragma once

#include "libbirch/FiberState.hpp"
#include "libbirch/Enter.hpp"
#include "libbirch/Clone.hpp"

namespace bi {
/**
 * Fiber.
 *
 * @ingroup libbirch
 *
 * @tparam YieldType Yield type.
 */
template<class YieldType>
class Fiber {
public:
  /**
   * Constructor.
   */
  Fiber(const std::shared_ptr<FiberState<YieldType>>& state = nullptr);

  /**
   * Copy constructor.
   */
  Fiber(const Fiber<YieldType>& o);

  /**
   * Move constructor.
   */
  Fiber(Fiber<YieldType> && o) = default;

  /**
   * Copy assignment.
   */
  Fiber<YieldType>& operator=(const Fiber<YieldType>& o);

  /**
   * Move assignment.
   */
  Fiber<YieldType>& operator=(Fiber<YieldType> && o) = default;

  /**
   * Run to next yield point.
   *
   * @return Was a value yielded?
   */
  bool query();

  /**
   * Get the last yield value.
   */
  YieldType& get();

private:
  /**
   * Mark a fiber as dirty, when it is copied.
   *
   * @internal This must be const as it is applied, via a const_cast, to the
   * argumet of the copy constructor and assignment operator.
   */
  void dirty() const;

  /**
   * Fiber state.
   */
  std::shared_ptr<FiberState<YieldType>> state;

  /**
   * Is the fiber dirty?
   */
  bool isDirty;
};
}

template<class YieldType>
bi::Fiber<YieldType>::Fiber(
    const std::shared_ptr<FiberState<YieldType>>& state) :
    state(state),
    isDirty(false) {
  //
}

template<class YieldType>
bi::Fiber<YieldType>::Fiber(const Fiber<YieldType>& o) :
    state(o.state) {
  dirty();
  o.dirty();
}

template<class YieldType>
bi::Fiber<YieldType>& bi::Fiber<YieldType>::operator=(
    const Fiber<YieldType>& o) {
  if (this != &o) {  // for self-assignment, needn't dirty
    state = o.state;
    dirty();
    o.dirty();
  }
  return *this;
}

template<class YieldType>
bool bi::Fiber<YieldType>::query() {
  bool result = false;
  if (state) {
    if (isDirty) {
      Clone clone;
      state = state->clone();
      isDirty = false;
    }
    Enter enter(state->getWorld());
    result = state->query();
    if (!result) {
      state.reset();  // fiber has finished, delete the state
    }
  }
  return result;
}

template<class YieldType>
YieldType& bi::Fiber<YieldType>::get() {
  assert(state);
  return state->get();
}

template<class YieldType>
void bi::Fiber<YieldType>::dirty() const {
  const_cast<Fiber<YieldType>*>(this)->isDirty = true;
}
