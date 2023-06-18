/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Transpose {
  BIRCH_UNARY_FORM(Transpose)
  BIRCH_UNARY_SIZE(Transpose)
  BIRCH_UNARY_EVAL(Transpose, transpose)
  BIRCH_UNARY_GRAD(Transpose, transpose_grad)
};

BIRCH_UNARY_TYPE(Transpose)
BIRCH_UNARY_CALL(Transpose, transpose)

}
