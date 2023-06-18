/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct SimulateBeta {
  BIRCH_BINARY_FORM(SimulateBeta)
  BIRCH_BINARY_SIZE(SimulateBeta)
  BIRCH_BINARY_EVAL(SimulateBeta, simulate_beta)
};

BIRCH_BINARY_TYPE(SimulateBeta)
BIRCH_BINARY_CALL(SimulateBeta, simulate_beta)

}
