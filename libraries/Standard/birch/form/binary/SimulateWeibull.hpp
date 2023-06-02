/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct SimulateWeibull {
  BIRCH_BINARY_FORM(SimulateWeibull)
};

BIRCH_BINARY_SIZE(SimulateWeibull)
BIRCH_BINARY(SimulateWeibull, numbirch::simulate_weibull)

template<class Left, class Right>
auto simulate_weibull(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::simulate_weibull(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(SimulateWeibull);
  }
}

}
