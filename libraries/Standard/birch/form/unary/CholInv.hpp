/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct CholInv {
  BIRCH_UNARY_FORM(CholInv, numbirch::cholinv)
  BIRCH_UNARY_GRAD(numbirch::cholinv_grad)
  BIRCH_FORM
};

template<class Middle>
auto cholinv(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(CholInv);
}

}
