/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct SimulateBinomial {
  BIRCH_BINARY_FORM(SimulateBinomial, numbirch::simulate_binomial)
  BIRCH_FORM
};

template<class Left, class Right>
auto simulate_binomial(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::simulate_binomial(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(SimulateBinomial);
  }
}

}
