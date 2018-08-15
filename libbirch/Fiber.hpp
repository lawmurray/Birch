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
  Fiber(const SharedPtr<FiberState<YieldType>>& state = nullptr);

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
   * Fiber state.
   */
  SharedPtr<FiberState<YieldType>> state;
};
}

template<class YieldType>
bi::Fiber<YieldType>::Fiber(
    const SharedPtr<FiberState<YieldType>>& state) :
    state(state) {
  //
}

template<class YieldType>
bool bi::Fiber<YieldType>::query() {
  bool result = false;
  if (state) {
    if (state->isShared() || state->isDirty()) {
      state->dirty();
      Clone clone;
      state = state->clone();
    }
    Enter enter(state->getWorld());
    result = state->query();
    if (!result) {
      state = nullptr;  // fiber has finished, delete the state
    }
  }
  return result;
}

template<class YieldType>
YieldType& bi::Fiber<YieldType>::get() {
  bi_assert_msg(state, "fiber handle undefined");
  return state->get();
}
