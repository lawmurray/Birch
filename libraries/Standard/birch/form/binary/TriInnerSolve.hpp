/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct TriInnerSolve {
  BIRCH_BINARY_FORM(TriInnerSolve)
};

BIRCH_BINARY_SIZE(TriInnerSolve)
BIRCH_BINARY(TriInnerSolve, numbirch::triinnersolve)
BIRCH_BINARY_GRAD(TriInnerSolve, numbirch::triinnersolve_grad)

template<class Left, class Right>
auto triinnersolve(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(TriInnerSolve);
}

}
