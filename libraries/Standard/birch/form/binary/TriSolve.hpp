/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct TriSolve {
  BIRCH_BINARY_FORM(TriSolve, numbirch::trisolve)
  BIRCH_BINARY_GRAD(numbirch::trisolve_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto trisolve(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(TriSolve);
}

}
