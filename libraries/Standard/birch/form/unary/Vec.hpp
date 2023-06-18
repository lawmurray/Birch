/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Vec {
  BIRCH_UNARY_FORM(Vec)
  BIRCH_UNARY_SIZE(Vec)
  BIRCH_UNARY_EVAL(Vec, vec)
  BIRCH_UNARY_GRAD(Vec, vec_grad)
};

BIRCH_UNARY_TYPE(Vec)
BIRCH_UNARY_CALL(Vec, vec)

}
