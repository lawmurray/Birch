/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct CholInv {
  BIRCH_UNARY_FORM(CholInv)
};

BIRCH_UNARY_SIZE(CholInv)
BIRCH_UNARY(CholInv, numbirch::cholinv)
BIRCH_UNARY_GRAD(CholInv, numbirch::cholinv_grad)

template<class Middle>
auto cholinv(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(CholInv);
}

}
