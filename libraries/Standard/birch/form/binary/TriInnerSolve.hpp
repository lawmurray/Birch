/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct TriInnerSolve {
  BIRCH_BINARY_FORM(TriInnerSolve)
};

BIRCH_BINARY_SIZE(TriInnerSolve)
BIRCH_BINARY(TriInnerSolve, triinnersolve)
BIRCH_BINARY_GRAD(TriInnerSolve, triinnersolve_grad)

}
