/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct SimulateGamma {
  BIRCH_BINARY_FORM(SimulateGamma)
  BIRCH_BINARY_SIZE(SimulateGamma)
  BIRCH_BINARY_EVAL(SimulateGamma, simulate_gamma)
};

BIRCH_BINARY_TYPE(SimulateGamma)
BIRCH_BINARY_CALL(SimulateGamma, simulate_gamma)

}
