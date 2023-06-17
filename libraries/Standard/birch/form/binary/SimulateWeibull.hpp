/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct SimulateWeibull {
  BIRCH_BINARY_FORM(SimulateWeibull)
};

BIRCH_BINARY_SIZE(SimulateWeibull)
BIRCH_BINARY(SimulateWeibull, simulate_weibull)

}
