/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<argument Left, argument Middle, argument Right>
struct MatrixSingle {
  BIRCH_TERNARY_FORM(MatrixSingle, R, C)
  BIRCH_TERNARY_EVAL(MatrixSingle, single, R, C)
  BIRCH_TERNARY_GRAD(MatrixSingle, single_grad, R, C)

  int rows() const {
    return R;
  }

  int columns() const {
    return C;
  }

  int length() const {
    return R;
  }

  int size() const {
    return R*C;
  }
};

BIRCH_TERNARY_TYPE(MatrixSingle)
BIRCH_TERNARY_CALL(MatrixSingle, single, R, C)

}
