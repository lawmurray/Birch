/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::simulate_gamma;

template<class Left, class Right>
struct SimulateGamma : public Binary<Left,Right> {
  template<class T, class U>
  SimulateGamma(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(simulate_gamma)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
SimulateGamma<Left,Right> simulate_gamma(const Left& l, const Right& r) {
  return SimulateGamma<Left,Right>(l, r);
}

}
