/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct CholSolve {
  BIRCH_BINARY_FORM(CholSolve)
};

BIRCH_BINARY_SIZE(CholSolve)
BIRCH_BINARY(CholSolve, cholsolve)
BIRCH_BINARY_GRAD(CholSolve, cholsolve_grad)

}
