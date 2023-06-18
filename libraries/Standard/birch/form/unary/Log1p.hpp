/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Log1p {
  BIRCH_UNARY_FORM(Log1p)
  BIRCH_UNARY_SIZE(Log1p)
  BIRCH_UNARY_EVAL(Log1p, log1p)
  BIRCH_UNARY_GRAD(Log1p, log1p_grad)
};

BIRCH_UNARY_TYPE(Log1p)
BIRCH_UNARY_CALL(Log1p, log1p)

}
