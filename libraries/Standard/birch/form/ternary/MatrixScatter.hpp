/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<argument Left, argument Middle, argument Right>
struct MatrixScatter {
  BIRCH_TERNARY_FORM(MatrixScatter, R, C)
  BIRCH_TERNARY_EVAL(MatrixScatter, scatter, R, C)
  BIRCH_TERNARY_GRAD(MatrixScatter, scatter_grad, R, C)

  int rows() const {
    return R;
  }

  int columns() const {
    return C;
  }

  int size() const {
    return R*C;
  }
};

BIRCH_TERNARY_TYPE(MatrixScatter)
BIRCH_TERNARY_CALL(MatrixScatter, scatter, R, C)

}
