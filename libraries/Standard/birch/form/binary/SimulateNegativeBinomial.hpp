/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct SimulateNegativeBinomial {
  BIRCH_BINARY_FORM(SimulateNegativeBinomial)
  BIRCH_BINARY_SIZE(SimulateNegativeBinomial)
  BIRCH_BINARY_EVAL(SimulateNegativeBinomial, simulate_negative_binomial)
};

BIRCH_BINARY_TYPE(SimulateNegativeBinomial)
BIRCH_BINARY_CALL(SimulateNegativeBinomial, simulate_negative_binomial)

}
