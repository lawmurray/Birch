/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct SimulateNegativeBinomial {
  BIRCH_BINARY_FORM(SimulateNegativeBinomial)
};

BIRCH_BINARY_SIZE(SimulateNegativeBinomial)
BIRCH_BINARY(SimulateNegativeBinomial, simulate_negative_binomial)

}
