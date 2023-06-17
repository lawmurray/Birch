/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct SimulateUniform {
  BIRCH_BINARY_FORM(SimulateUniform)
};

BIRCH_BINARY_SIZE(SimulateUniform)
BIRCH_BINARY(SimulateUniform, simulate_uniform)

}
