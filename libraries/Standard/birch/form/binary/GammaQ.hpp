/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"
#include "birch/form/Nullary.hpp"

namespace birch {

template<argument Left, argument Right>
struct GammaQ {
  BIRCH_BINARY_FORM(GammaQ)
  BIRCH_BINARY_SIZE(GammaQ)
  BIRCH_BINARY_EVAL(GammaQ, gamma_q)
  BIRCH_NULLARY_GRAD(GammaQ)
};

BIRCH_BINARY_TYPE(GammaQ)
BIRCH_BINARY_CALL(GammaQ, gamma_q)

}
