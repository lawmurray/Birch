/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct TriInnerSelf {
  BIRCH_UNARY_FORM(TriInnerSelf)
  BIRCH_UNARY_SIZE(TriInnerSelf)
  BIRCH_UNARY_EVAL(TriInnerSelf, triinner)
  BIRCH_UNARY_GRAD(TriInnerSelf, triinner_grad)
};

BIRCH_UNARY_TYPE(TriInnerSelf)
BIRCH_UNARY_CALL(TriInnerSelf, triinner)

}
