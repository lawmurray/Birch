/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct LGamma {
  BIRCH_UNARY_FORM(LGamma)
};

BIRCH_UNARY_SIZE(LGamma)
BIRCH_UNARY(LGamma, lgamma)
BIRCH_UNARY_GRAD(LGamma, lgamma_grad)

}
