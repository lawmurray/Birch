/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct LDet {
  BIRCH_UNARY_FORM(LDet)
  BIRCH_UNARY_SIZE(LDet)
  BIRCH_UNARY_EVAL(LDet, ldet)
  BIRCH_UNARY_GRAD(LDet, ldet_grad)
};

BIRCH_UNARY_TYPE(LDet)
BIRCH_UNARY_CALL(LDet, ldet)

}
