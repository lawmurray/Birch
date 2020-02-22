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
 * @tparam Yield Yield type.
 * @tparam Return Return type.
 */
template<class Yield, class Return>
class Fiber {
public:
  using yield_type = Yield;
  using return_type = Return;
  using state_type = Lazy<SharedPtr<FiberState<Yield,Return>>>;

  /**
   * Constructor.
   */
  Fiber() {
    //
  }

  /**
   * Constructor.
   */
  Fiber(const state_type& state) :
      state(state) {
    //
  }

  /**
   * Constructor.
   */
  template<class T, std::enable_if_t<std::is_same<T,yield_type>::value &&
      !std::is_void<yield_type>::value,int> = 0>
  Fiber(const T& yieldValue, const state_type& state) :
      yieldValue(yieldValue),
      state(state) {
    //
  }

  /**
   * Constructor.
   */
  template<class T, std::enable_if_t<std::is_same<T,return_type>::value &&
      !std::is_void<return_type>::value,int> = 0>
  Fiber(const T& returnValue) :
      returnValue(returnValue) {
    //
  }

  /**
   * Clone the fiber.
   */
  Fiber<Yield,Return> clone() const {
    return Fiber(*this);
  }

  /**
   * Run to next yield point.
   *
   * @return Was a value yielded?
   */
  bool query() {
    if (state.query()) {
      *this = state.get()->query();
    }
    return yieldValue.query();
  }

  /**
   * Get the current yield value.
   */
  auto get() {
    return yieldValue.get();
  }

private:
  /**
   * Yield value.
   */
  Optional<yield_type> yieldValue;

  /**
   * Return value.
   */
  Optional<return_type> returnValue;

  /**
   * Fiber state.
   */
  Optional<state_type> state;
};

template<class Yield, class Return>
struct is_value<Fiber<Yield,Return>> {
  static const bool value = false;
};
}
