/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct SimulateBinomial {
  BIRCH_BINARY_FORM(SimulateBinomial)
};

BIRCH_BINARY_SIZE(SimulateBinomial)
BIRCH_BINARY(SimulateBinomial, simulate_binomial)

}
