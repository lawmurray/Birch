/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Chol {
  BIRCH_UNARY_FORM(Chol, numbirch::chol)
  BIRCH_UNARY_GRAD(numbirch::chol_grad)
  BIRCH_FORM
};

template<class Middle>
auto chol(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(Chol);
}

}
