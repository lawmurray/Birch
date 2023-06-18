/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct OuterSelf {
  BIRCH_UNARY_FORM(OuterSelf)
  BIRCH_UNARY_SIZE(OuterSelf)
  BIRCH_UNARY_EVAL(OuterSelf, outer)
  BIRCH_UNARY_GRAD(OuterSelf, outer_grad)
};

BIRCH_UNARY_TYPE(OuterSelf)
BIRCH_UNARY_CALL(OuterSelf, outer)

}
