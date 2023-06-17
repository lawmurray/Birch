/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Tanh {
  BIRCH_UNARY_FORM(Tanh)
};

BIRCH_UNARY_SIZE(Tanh)
BIRCH_UNARY(Tanh, tanh)
BIRCH_UNARY_GRAD(Tanh, tanh_grad)

}
