/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct CholSolve {
  BIRCH_BINARY_FORM(CholSolve)
  BIRCH_BINARY_SIZE(CholSolve)
  BIRCH_BINARY_EVAL(CholSolve, cholsolve)
  BIRCH_BINARY_GRAD_WITH_RESULT(CholSolve, cholsolve_grad)
};

BIRCH_BINARY_TYPE(CholSolve)
BIRCH_BINARY_CALL(CholSolve, cholsolve)

}
