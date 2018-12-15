/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
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
  Fiber(const SharedCOW<FiberState<YieldType>>& state = nullptr);

  /**
   * Clone the fiber.
   */
  Fiber<YieldType> clone() const;

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

public:
  /**
   * Fiber state.
   */
  SharedCOW<FiberState<YieldType>> state;
};
}

template<class YieldType>
bi::Fiber<YieldType>::Fiber(
    const SharedCOW<FiberState<YieldType>>& state) :
    state(state) {
  //
}

template<class YieldType>
bi::Fiber<YieldType> bi::Fiber<YieldType>::clone() const {
  return Fiber<YieldType>(state.clone());
}

template<class YieldType>
bool bi::Fiber<YieldType>::query() {
  bool result = false;
  if (state.query()) {
    result = state->query();
    if (!result) {
      state = nullptr;  // fiber has finished, delete the state
    }
  }
  return result;
}

template<class YieldType>
YieldType& bi::Fiber<YieldType>::get() {
  bi_assert_msg(state.query(), "fiber handle undefined");
  return state->get();
}
