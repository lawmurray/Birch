/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Rectify {
  BIRCH_UNARY_FORM(Rectify)
  BIRCH_UNARY_SIZE(Rectify)
  BIRCH_UNARY_EVAL(Rectify, rectify)
  BIRCH_UNARY_GRAD(Rectify, rectify_grad)
};

BIRCH_UNARY_TYPE(Rectify)
BIRCH_UNARY_CALL(Rectify, rectify)

}
