/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Chol {
  BIRCH_UNARY_FORM(Chol)
};

BIRCH_UNARY_SIZE(Chol)
BIRCH_UNARY(Chol, numbirch::chol)
BIRCH_UNARY_GRAD(Chol, numbirch::chol_grad)

template<class Middle>
auto chol(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(Chol);
}

}
