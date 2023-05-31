/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct SimulateBeta {
  BIRCH_BINARY_FORM(SimulateBeta, numbirch::simulate_beta)
  BIRCH_FORM
};

template<class Left, class Right>
auto simulate_beta(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::simulate_beta(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(SimulateBeta);
  }
}

}
