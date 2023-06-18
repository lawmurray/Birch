/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct LBeta {
  BIRCH_BINARY_FORM(LBeta)
  BIRCH_BINARY_SIZE(LBeta)
  BIRCH_BINARY_EVAL(LBeta, lbeta)
  BIRCH_BINARY_GRAD(LBeta, lbeta_grad)
};

BIRCH_BINARY_TYPE(LBeta)
BIRCH_BINARY_CALL(LBeta, lbeta)

}
