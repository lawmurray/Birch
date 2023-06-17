/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct SimulateGaussian {
  BIRCH_BINARY_FORM(SimulateGaussian)
};

BIRCH_BINARY_SIZE(SimulateGaussian)
BIRCH_BINARY(SimulateGaussian, simulate_gaussian)

}
