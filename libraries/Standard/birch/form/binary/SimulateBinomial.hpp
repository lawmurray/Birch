/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::simulate_binomial;

template<class Left, class Right>
struct SimulateBinomial : public Binary<Left,Right> {
  template<class T, class U>
  SimulateBinomial(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(simulate_binomial)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
SimulateBinomial<Left,Right> simulate_binomial(const Left& l, const Right& r) {
  return SimulateBinomial<Left,Right>(l, r);
}

}
