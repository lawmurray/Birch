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
 * @tparam YieldType Yield type.
 */
template<class YieldType>
class Fiber {
public:
  /**
   * Constructor.
   */
  Fiber(const Shared<FiberState<YieldType>>& state = nullptr);

  /**
   * Clone the fiber.
   */
  Fiber<YieldType> clone();

  /**
   * Freeze the fiber.
   */
  void freeze();

  /**
   * Get the context of the fiber state.
   */
  Memo* getContext() const;

  /**
   * Run to next yield point.
   *
   * @return Was a value yielded?
   */
  bool query() const;

  /**
   * Get the last yield value.
   */
  YieldType get() const;

public:
  /**
   * Fiber state.
   */
  Shared<FiberState<YieldType>> state;
};
}

template<class YieldType>
bi::Fiber<YieldType>::Fiber(
    const Shared<FiberState<YieldType>>& state) :
    state(state) {
  //
}

template<class YieldType>
bi::Fiber<YieldType> bi::Fiber<YieldType>::clone() {
  return Fiber<YieldType>(state.clone());
}

template<class YieldType>
void bi::Fiber<YieldType>::freeze() {
  state.freeze();
}

template<class YieldType>
bi::Memo* bi::Fiber<YieldType>::getContext() const {
  return state.getContext();
}

template<class YieldType>
bool bi::Fiber<YieldType>::query() const {
  bool result = false;
  if (state.query()) {
    result = state->query();
    if (!result) {
      const_cast<Fiber<YieldType>*>(this)->state = nullptr;  // fiber has finished, delete the state
    }
  }
  return result;
}

template<class YieldType>
YieldType bi::Fiber<YieldType>::get() const {
  bi_assert_msg(state.query(), "fiber handle undefined");
  return state->get();
}
