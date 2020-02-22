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

  Fiber(const Fiber&) = default;
  Fiber(Fiber&&) = default;
  Fiber& operator=(const Fiber&) = default;
  Fiber& operator=(Fiber&&) = default;

  /**
   * Constructor. Used:
   *
   * @li for an uninitialized fiber handle, and
   * @li for returns in fibers with a return type of `void`, where no state
   * or resume function is required, and no value is returned.
   */
  Fiber() {
    //
  }

  /**
   * Constructor. Used:
   *
   * @li in the initialization function of all fibers, where a state and
   * start function are required, but no value is yielded, and
   * @li for yields in fibers with a yield type of `void`, where a state and
   * resume function are required, but no value is yielded.
   */
  Fiber(const state_type& state) :
      state(state) {
    //
  }

  /**
   * Constructor. Used for yields in fibers with a yield type that is not
   * `void`, where a state and resume function are required, along with a
   * yield value.
   */
  template<class T, std::enable_if_t<std::is_same<T,yield_type>::value &&
      !std::is_void<yield_type>::value,int> = 0>
  Fiber(const T& yieldValue, const state_type& state) :
      yieldValue(yieldValue),
      state(state) {
    //
  }

  /**
   * Constructor. Used for returns in fibers with a return type that is not
   * `void`, where a state and resume function are not required, and a value
   * is returned.
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
