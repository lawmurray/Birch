/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct LGammaP {
  BIRCH_BINARY_FORM(LGammaP)
  BIRCH_BINARY_SIZE(LGammaP)
  BIRCH_BINARY_EVAL(LGammaP, lgamma)
  BIRCH_BINARY_GRAD(LGammaP, lgamma_grad)
};

BIRCH_BINARY_TYPE(LGammaP)
BIRCH_BINARY_CALL(LGammaP, lgamma)

}
