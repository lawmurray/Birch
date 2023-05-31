/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct VectorScatter {
  BIRCH_BINARY_FORM(VectorScatter, numbirch::scatter, n)
  BIRCH_BINARY_GRAD(numbirch::scatter_grad, n)
  BIRCH_FORM
};

template<class Left, class Right>
auto scatter(const Left& l, const Right& r, const int n) {
  return BIRCH_BINARY_CONSTRUCT(VectorScatter, n);
}

}
