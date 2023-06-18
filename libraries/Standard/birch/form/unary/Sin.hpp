/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Sin {
  BIRCH_UNARY_FORM(Sin)
  BIRCH_UNARY_SIZE(Sin)
  BIRCH_UNARY_EVAL(Sin, sin)
  BIRCH_UNARY_GRAD(Sin, sin_grad)
};

BIRCH_UNARY_TYPE(Sin)
BIRCH_UNARY_CALL(Sin, sin)

}
