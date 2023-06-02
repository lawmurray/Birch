/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct SimulateNegativeBinomial {
  BIRCH_BINARY_FORM(SimulateNegativeBinomial)
};

BIRCH_BINARY_SIZE(SimulateNegativeBinomial)
BIRCH_BINARY(SimulateNegativeBinomial, numbirch::simulate_negative_binomial)

template<class Left, class Right>
auto simulate_negative_binomial(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::simulate_negative_binomial(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(SimulateNegativeBinomial);
  }
}

}
