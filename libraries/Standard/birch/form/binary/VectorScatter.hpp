/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct VectorScatter {
  BIRCH_BINARY_FORM(VectorScatter, n)
  BIRCH_BINARY_EVAL(VectorScatter, scatter, n)
  BIRCH_BINARY_GRAD(VectorScatter, scatter_grad, n)

  int rows() const {
    return n;
  }

  static constexpr int columns() {
    return 1;
  }
};

BIRCH_BINARY_TYPE(VectorScatter)
BIRCH_BINARY_CALL(VectorScatter, scatter, n)

}
