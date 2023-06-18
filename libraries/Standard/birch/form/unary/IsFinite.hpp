/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct IsFinite {
  BIRCH_UNARY_FORM(IsFinite)
  BIRCH_UNARY_SIZE(IsFinite)
  BIRCH_UNARY_EVAL(IsFinite, isfinite)
  BIRCH_UNARY_GRAD(IsFinite, isfinite_grad)
};

BIRCH_UNARY_TYPE(IsFinite)
BIRCH_UNARY_CALL(IsFinite, isfinite)

}
