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
   * Constructor.
   */
  Fiber() {
    //
  }

  /**
   * Constructor.
   */
  Fiber(Label* context, const Shared<FiberState<YieldType>>& state) :
      state(context, state) {
    //
  }

  /**
   * Copy constructor.
   */
  Fiber(Label* context, const Fiber<YieldType>& o) :
      state(context, o.state) {
    //
  }

  /**
   * Move constructor.
   */
  Fiber(Label* context, Fiber<YieldType>&& o) :
      state(context, std::move(o.state)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  Fiber(Label* context, Label* label, const Fiber<YieldType>& o) :
      state(context, label, o.state) {
    //
  }

  /**
   * Copy assignment.
   */
  Fiber& assign(Label* context, const Fiber<YieldType>& o) {
    state.assign(context, o.state);
    return *this;
  }

  /**
   * Move assignment.
   */
  Fiber& assign(Label* context, Fiber<YieldType>&& o) {
    state.assign(context, std::move(o.state));
    return *this;
  }

  /**
   * Clone the fiber.
   */
  Fiber<YieldType> clone() const {
    return Fiber<YieldType>(state.clone());
  }

  /**
   * Freeze the fiber.
   */
  void freeze() const {
    state.freeze();
  }

  /**
   * Thaw the fiber.
   */
  void thaw(LazyLabel* label) const {
    state.thaw(label);
  }

  /**
   * Finish the fiber.
   */
  void finish() const {
    state.finish();
  }

  /**
   * Run to next yield point.
   *
   * @return Was a value yielded?
   */
  bool query() const {
    bool result = false;
    if (state.query()) {
      result = state->query();
      if (!result) {
        const_cast<Fiber<YieldType>*>(this)->state.release();
        // ^ fiber has finished, delete the state
      }
    }
    return result;
  }


  /**
   * Get the last yield value.
   */
  YieldType get() const {
    libbirch_assert_msg_(state.query(), "fiber handle undefined");
    return state->get();
  }


private:
  /**
   * Fiber state.
   */
  Shared<FiberState<YieldType>> state;
};

template<class T>
struct is_value<Fiber<T>> {
  static const bool value = false;
};
}
