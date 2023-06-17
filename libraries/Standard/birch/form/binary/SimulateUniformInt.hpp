/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct SimulateUniformInt {
  BIRCH_BINARY_FORM(SimulateUniformInt)
};

BIRCH_BINARY_SIZE(SimulateUniformInt)
BIRCH_BINARY(SimulateUniformInt, simulate_uniform_int)

}
