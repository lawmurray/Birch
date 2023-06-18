/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct SimulateBinomial {
  BIRCH_BINARY_FORM(SimulateBinomial)
  BIRCH_BINARY_SIZE(SimulateBinomial)
  BIRCH_BINARY_EVAL(SimulateBinomial, simulate_binomial)
};

BIRCH_BINARY_TYPE(SimulateBinomial)
BIRCH_BINARY_CALL(SimulateBinomial, simulate_binomial)

}
