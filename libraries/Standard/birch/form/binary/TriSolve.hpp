/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct TriSolve {
  BIRCH_BINARY_FORM(TriSolve)
};

BIRCH_BINARY_SIZE(TriSolve)
BIRCH_BINARY(TriSolve, trisolve)
BIRCH_BINARY_GRAD(TriSolve, trisolve_grad)

}
