/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Transpose {
  BIRCH_UNARY_FORM(Transpose)
};

BIRCH_UNARY_SIZE(Transpose)
BIRCH_UNARY(Transpose, transpose)
BIRCH_UNARY_GRAD(Transpose, transpose_grad)

}
