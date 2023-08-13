/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Max {
  BIRCH_UNARY_FORM(Max)
  BIRCH_UNARY_SIZE(Max)
  BIRCH_UNARY_EVAL(Max, max)
  BIRCH_UNARY_GRAD_WITH_RESULT(Max, max_grad)
};

BIRCH_UNARY_TYPE(Max)
BIRCH_UNARY_CALL(Max, max)

}
