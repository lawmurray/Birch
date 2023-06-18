/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct DotSelf {
  BIRCH_UNARY_FORM(DotSelf)
  BIRCH_UNARY_SIZE(DotSelf)
  BIRCH_UNARY_EVAL(DotSelf, dot)
  BIRCH_UNARY_GRAD(DotSelf, dot_grad)
};

BIRCH_UNARY_TYPE(DotSelf)
BIRCH_UNARY_CALL(DotSelf, dot)

}
