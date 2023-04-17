/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::simulate_gaussian;

template<class Left, class Right>
struct SimulateGaussian : public Binary<Left,Right> {
  template<class T, class U>
  SimulateGaussian(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(simulate_gaussian)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
SimulateGaussian<Left,Right> simulate_gaussian(const Left& l, const Right& r) {
  return SimulateGaussian<Left,Right>(l, r);
}

}
