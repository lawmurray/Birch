/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct TriInnerSolve {
  BIRCH_BINARY_FORM(TriInnerSolve, numbirch::triinnersolve)
  BIRCH_BINARY_GRAD(numbirch::triinnersolve_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto triinnersolve(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(TriInnerSolve);
}

}
