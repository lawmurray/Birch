/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct TriInv {
  BIRCH_UNARY_FORM(TriInv, numbirch::triinv)
  BIRCH_UNARY_GRAD(numbirch::triinv_grad)
  BIRCH_FORM
};

template<class Middle>
auto triinv(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(TriInv);
}

}
