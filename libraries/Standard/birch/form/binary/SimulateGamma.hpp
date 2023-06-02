/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct SimulateGamma {
  BIRCH_BINARY_FORM(SimulateGamma)
};

BIRCH_BINARY_SIZE(SimulateGamma)
BIRCH_BINARY(SimulateGamma, numbirch::simulate_gamma)

template<class Left, class Right>
auto simulate_gamma(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::simulate_gamma(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(SimulateGamma);
  }
}

}
