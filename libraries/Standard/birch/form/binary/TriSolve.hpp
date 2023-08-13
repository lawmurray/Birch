/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct TriSolve {
  BIRCH_BINARY_FORM(TriSolve)
  BIRCH_BINARY_SIZE(TriSolve)
  BIRCH_BINARY_EVAL(TriSolve, trisolve)
  BIRCH_BINARY_GRAD_WITH_RESULT(TriSolve, trisolve_grad)
};

BIRCH_BINARY_TYPE(TriSolve)
BIRCH_BINARY_CALL(TriSolve, trisolve)

}
