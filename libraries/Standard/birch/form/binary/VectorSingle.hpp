/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct VectorSingle {
  BIRCH_BINARY_FORM(VectorSingle, n)
  BIRCH_BINARY_EVAL(VectorSingle, single, n)
  BIRCH_BINARY_GRAD(VectorSingle, single_grad, n)

  int rows() const {
    return n;
  }

  static constexpr int columns() {
    return 1;
  }
};

BIRCH_BINARY_TYPE(VectorSingle)
BIRCH_BINARY_CALL(VectorSingle, single, n)

}
