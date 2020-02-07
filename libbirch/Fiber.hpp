/**
 * @file
 */
#pragma once

#include "libbirch/FiberOutput.hpp"

namespace libbirch {
/**
 * Fiber.
 *
 * @ingroup libbirch
 *
 * @tparam YieldType Yield type.
 * @tparam ReturnType Return type.
 */
template<class YieldType, class ReturnType>
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
  Fiber(Label* context) {
    //
  }

  /**
   * Constructor.
   */
  Fiber(Label* context,
      const Lazy<SharedPtr<FiberOutput<YieldType,ReturnType>>>& state) :
      state(context, state) {
    //
  }

  /**
   * Copy constructor.
   */
  Fiber(Label* context, const Fiber<YieldType,ReturnType>& o) :
      state(context, o.state) {
    //
  }

  /**
   * Move constructor.
   */
  Fiber(Label* context, Fiber<YieldType,ReturnType>&& o) :
      state(context, std::move(o.state)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  Fiber(Label* context, Label* label, const Fiber<YieldType,ReturnType>& o) :
      state(context, label, o.state) {
    //
  }

  /**
   * Copy assignment.
   */
  Fiber& assign(Label* context, const Fiber<YieldType,ReturnType>& o) {
    state.assign(context, o.state);
    return *this;
  }

  /**
   * Move assignment.
   */
  Fiber& assign(Label* context, Fiber<YieldType,ReturnType>&& o) {
    state.assign(context, std::move(o.state));
    return *this;
  }

  /**
   * Clone the fiber.
   */
  Fiber<YieldType,ReturnType> clone(Label* context) const {
    return Fiber<YieldType,ReturnType>(context, state.clone(context));
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
  void thaw(Label* label) const {
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
        const_cast<Fiber<YieldType,ReturnType>*>(this)->state.release();
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
  Lazy<SharedPtr<FiberOutput<YieldType,ReturnType>>> state;
};

template<class T, class U>
struct is_value<Fiber<T,U>> {
  static const bool value = false;
};
}
