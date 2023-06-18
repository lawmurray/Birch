/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Min {
  BIRCH_UNARY_FORM(Min)
  BIRCH_UNARY_SIZE(Min)
  BIRCH_UNARY_EVAL(Min, min)
  BIRCH_UNARY_GRAD(Min, min_grad)
};

BIRCH_UNARY_TYPE(Min)
BIRCH_UNARY_CALL(Min, min)

}
