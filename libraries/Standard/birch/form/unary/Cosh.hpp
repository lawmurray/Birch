/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Cosh {
  BIRCH_UNARY_FORM(Cosh)
  BIRCH_UNARY_SIZE(Cosh)
  BIRCH_UNARY_EVAL(Cosh, cosh)
  BIRCH_UNARY_GRAD(Cosh, cosh_grad)
};

BIRCH_UNARY_TYPE(Cosh)
BIRCH_UNARY_CALL(Cosh, cosh)

}
