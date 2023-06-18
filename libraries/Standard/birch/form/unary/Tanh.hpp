/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Tanh {
  BIRCH_UNARY_FORM(Tanh)
  BIRCH_UNARY_SIZE(Tanh)
  BIRCH_UNARY_EVAL(Tanh, tanh)
  BIRCH_UNARY_GRAD(Tanh, tanh_grad)
};

BIRCH_UNARY_TYPE(Tanh)
BIRCH_UNARY_CALL(Tanh, tanh)

}
