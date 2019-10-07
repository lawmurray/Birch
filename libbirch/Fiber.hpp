/**
 * @file
 */
#pragma once

#include "libbirch/FiberState.hpp"

namespace libbirch {
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
   * Default constructor.
   */
  Fiber();

  /**
   * Constructor.
   */
  Fiber(const Shared<FiberState<YieldType>>& state);

  /**
   * Clone the fiber.
   */
  Fiber<YieldType> clone() const;

  /**
   * Freeze the fiber.
   */
  void freeze() const;

  /**
   * Thaw the fiber.
   */
  void thaw(LazyLabel* label) const;

  /**
   * Finish the fiber.
   */
  void finish() const;

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

private:
  /**
   * Fiber state.
   */
  Shared<FiberState<YieldType>> state;
};
}

template<class YieldType>
libbirch::Fiber<YieldType>::Fiber() {
  //
}

template<class YieldType>
libbirch::Fiber<YieldType>::Fiber(
    const Shared<FiberState<YieldType>>& state) :
    state(state) {
  //
}

template<class YieldType>
libbirch::Fiber<YieldType> libbirch::Fiber<YieldType>::clone() const {
  return Fiber<YieldType>(state.clone());
}

template<class YieldType>
void libbirch::Fiber<YieldType>::freeze() const {
  state.freeze();
}

template<class YieldType>
void libbirch::Fiber<YieldType>::finish() const {
  state.finish();
}

template<class YieldType>
void libbirch::Fiber<YieldType>::thaw(LazyLabel* label) const {
  state.thaw(label);
}

template<class YieldType>
bool libbirch::Fiber<YieldType>::query() const {
  bool result = false;
  if (state.query()) {
    result = state->query();
    if (!result) {
      const_cast<Fiber<YieldType>*>(this)->state = nullptr;
      // ^ fiber has finished, delete the state
    }
  }
  return result;
}

template<class YieldType>
YieldType libbirch::Fiber<YieldType>::get() const {
  libbirch_assert_msg_(state.query(), "fiber handle undefined");
  return state->get();
}
