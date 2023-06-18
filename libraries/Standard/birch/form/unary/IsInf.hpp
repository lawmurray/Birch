/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct IsInf {
  BIRCH_UNARY_FORM(IsInf)
  BIRCH_UNARY_SIZE(IsInf)
  BIRCH_UNARY_EVAL(IsInf, isinf)
  BIRCH_UNARY_GRAD(IsInf, isinf_grad)
};

BIRCH_UNARY_TYPE(IsInf)
BIRCH_UNARY_CALL(IsInf, isinf)

}
