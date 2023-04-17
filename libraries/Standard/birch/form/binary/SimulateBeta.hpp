/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::simulate_beta;

template<class Left, class Right>
struct SimulateBeta : public Binary<Left,Right> {
  template<class T, class U>
  SimulateBeta(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(simulate_beta)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
SimulateBeta<Left,Right> simulate_beta(const Left& l, const Right& r) {
  return SimulateBeta<Left,Right>(l, r);
}

}
