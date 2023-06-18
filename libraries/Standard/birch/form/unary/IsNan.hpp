/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct IsNan {
  BIRCH_UNARY_FORM(IsNan)
  BIRCH_UNARY_SIZE(IsNan)
  BIRCH_UNARY_EVAL(IsNan, isnan)
  BIRCH_UNARY_GRAD(IsNan, isnan_grad)
};

BIRCH_UNARY_TYPE(IsNan)
BIRCH_UNARY_CALL(IsNan, isnan)

}
