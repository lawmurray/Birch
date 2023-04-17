/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::simulate_weibull;

template<class Left, class Right>
struct SimulateWeibull : public Binary<Left,Right> {
  template<class T, class U>
  SimulateWeibull(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(simulate_weibull)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
SimulateWeibull<Left,Right> simulate_weibull(const Left& l, const Right& r) {
  return SimulateWeibull<Left,Right>(l, r);
}

}
