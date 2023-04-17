/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::simulate_negative_binomial;

template<class Left, class Right>
struct SimulateNegativeBinomial : public Binary<Left,Right> {
  template<class T, class U>
  SimulateNegativeBinomial(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(simulate_negative_binomial)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
SimulateNegativeBinomial<Left,Right> simulate_negative_binomial(const Left& l, const Right& r) {
  return SimulateNegativeBinomial<Left,Right>(l, r);
}

}
