/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Floor {
  BIRCH_UNARY_FORM(Floor)
  BIRCH_UNARY_SIZE(Floor)
  BIRCH_UNARY_EVAL(Floor, floor)
  BIRCH_UNARY_GRAD(Floor, floor_grad)
};

BIRCH_UNARY_TYPE(Floor)
BIRCH_UNARY_CALL(Floor, floor)

}
