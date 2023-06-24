/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"
#include "birch/form/Nullary.hpp"

namespace birch {

template<argument Left, argument Right>
struct GammaP {
  BIRCH_BINARY_FORM(GammaP)
  BIRCH_BINARY_SIZE(GammaP)
  BIRCH_BINARY_EVAL(GammaP, gamma_p)
  BIRCH_NULLARY_GRAD(GammaP)
};

BIRCH_BINARY_TYPE(GammaP)
BIRCH_BINARY_CALL(GammaP, gamma_p)

}
