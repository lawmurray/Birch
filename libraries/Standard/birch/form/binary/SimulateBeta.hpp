/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct SimulateBeta {
  BIRCH_BINARY_FORM(SimulateBeta)
};

BIRCH_BINARY_SIZE(SimulateBeta)
BIRCH_BINARY(SimulateBeta, simulate_beta)

}
