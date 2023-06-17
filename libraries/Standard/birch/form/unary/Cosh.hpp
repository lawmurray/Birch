/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Cosh {
  BIRCH_UNARY_FORM(Cosh)
};

BIRCH_UNARY_SIZE(Cosh)
BIRCH_UNARY(Cosh, cosh)
BIRCH_UNARY_GRAD(Cosh, cosh_grad)

}
