/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct VectorScatter {
  BIRCH_BINARY_FORM(VectorScatter, n)
};

BIRCH_BINARY_SIZE(VectorScatter)
BIRCH_BINARY(VectorScatter, numbirch::scatter, n)
BIRCH_BINARY_GRAD(VectorScatter, numbirch::scatter_grad, n)

template<class Left, class Right>
auto scatter(const Left& l, const Right& r, const int n) {
  return BIRCH_BINARY_CONSTRUCT(VectorScatter, n);
}

}
