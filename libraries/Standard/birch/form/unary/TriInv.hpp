/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct TriInv {
  BIRCH_UNARY_FORM(TriInv)
};

BIRCH_UNARY_SIZE(TriInv)
BIRCH_UNARY(TriInv, numbirch::triinv)
BIRCH_UNARY_GRAD(TriInv, numbirch::triinv_grad)

template<class Middle>
auto triinv(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(TriInv);
}

}
