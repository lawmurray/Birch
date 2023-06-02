/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct CholSolve {
  BIRCH_BINARY_FORM(CholSolve)
};

BIRCH_BINARY_SIZE(CholSolve)
BIRCH_BINARY(CholSolve, numbirch::cholsolve)
BIRCH_BINARY_GRAD(CholSolve, numbirch::cholsolve_grad)

template<class Left, class Right>
auto cholsolve(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(CholSolve);
}

}
