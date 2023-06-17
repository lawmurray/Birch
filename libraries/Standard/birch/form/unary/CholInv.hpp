/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct CholInv {
  BIRCH_UNARY_FORM(CholInv)
};

BIRCH_UNARY_SIZE(CholInv)
BIRCH_UNARY(CholInv, cholinv)
BIRCH_UNARY_GRAD(CholInv, cholinv_grad)

}
