/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct TriInnerSolve {
  BIRCH_BINARY_FORM(TriInnerSolve)
  BIRCH_BINARY_SIZE(TriInnerSolve)
  BIRCH_BINARY_EVAL(TriInnerSolve, triinnersolve)
  BIRCH_BINARY_GRAD_WITH_RESULT(TriInnerSolve, triinnersolve_grad)
};

BIRCH_BINARY_TYPE(TriInnerSolve)
BIRCH_BINARY_CALL(TriInnerSolve, triinnersolve)

}
