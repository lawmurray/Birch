/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Atan {
  BIRCH_UNARY_FORM(Atan)
  BIRCH_UNARY_SIZE(Atan)
  BIRCH_UNARY_EVAL(Atan, atan)
  BIRCH_UNARY_GRAD(Atan, atan_grad)
};

BIRCH_UNARY_TYPE(Atan)
BIRCH_UNARY_CALL(Atan, atan)

}
