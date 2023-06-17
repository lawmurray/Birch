/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Sqrt {
  BIRCH_UNARY_FORM(Sqrt)
};

BIRCH_UNARY_SIZE(Sqrt)
BIRCH_UNARY(Sqrt, sqrt)
BIRCH_UNARY_GRAD(Sqrt, sqrt_grad)

}
