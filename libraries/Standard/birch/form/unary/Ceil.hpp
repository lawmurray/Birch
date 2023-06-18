/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Ceil {
  BIRCH_UNARY_FORM(Ceil)
  BIRCH_UNARY_SIZE(Ceil)
  BIRCH_UNARY_EVAL(Ceil, ceil)
  BIRCH_UNARY_GRAD(Ceil, ceil_grad)
};

BIRCH_UNARY_TYPE(Ceil)
BIRCH_UNARY_CALL(Ceil, ceil)

}
