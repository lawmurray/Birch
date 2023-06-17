/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct LTriDet {
  BIRCH_UNARY_FORM(LTriDet)
};

BIRCH_UNARY_SIZE(LTriDet)
BIRCH_UNARY(LTriDet, ltridet)
BIRCH_UNARY_GRAD(LTriDet, ltridet_grad)

}
