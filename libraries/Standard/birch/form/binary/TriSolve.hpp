/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct TriSolve {
  BIRCH_BINARY_FORM(TriSolve)
};

BIRCH_BINARY_SIZE(TriSolve)
BIRCH_BINARY(TriSolve, numbirch::trisolve)
BIRCH_BINARY_GRAD(TriSolve, numbirch::trisolve_grad)

template<class Left, class Right>
auto trisolve(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(TriSolve);
}

}
