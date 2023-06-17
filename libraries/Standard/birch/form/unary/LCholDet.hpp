/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct LCholDet {
  BIRCH_UNARY_FORM(LCholDet)
};

BIRCH_UNARY_SIZE(LCholDet)
BIRCH_UNARY(LCholDet, lcholdet)
BIRCH_UNARY_GRAD(LCholDet, lcholdet_grad)

}
