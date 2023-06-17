/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Atan {
  BIRCH_UNARY_FORM(Atan)
};

BIRCH_UNARY_SIZE(Atan)
BIRCH_UNARY(Atan, atan)
BIRCH_UNARY_GRAD(Atan, atan_grad)

}
