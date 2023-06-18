/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct CholInv {
  BIRCH_UNARY_FORM(CholInv)
  BIRCH_UNARY_SIZE(CholInv)
  BIRCH_UNARY_EVAL(CholInv, cholinv)
  BIRCH_UNARY_GRAD(CholInv, cholinv_grad)
};

BIRCH_UNARY_TYPE(CholInv)
BIRCH_UNARY_CALL(CholInv, cholinv)

}
