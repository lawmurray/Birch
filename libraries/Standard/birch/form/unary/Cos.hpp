/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Cos {
  BIRCH_UNARY_FORM(Cos)
};

BIRCH_UNARY_SIZE(Cos)
BIRCH_UNARY(Cos, cos)
BIRCH_UNARY_GRAD(Cos, cos_grad)

}
