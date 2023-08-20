/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct MatrixFill {
  BIRCH_UNARY_FORM(MatrixFill, R, C)
  BIRCH_UNARY_EVAL(MatrixFill, fill, R, C)
  BIRCH_UNARY_GRAD(MatrixFill, fill_grad, R, C)

  int rows() const {
    return R;
  }

  int columns() const {
    return C;
  }
};

BIRCH_UNARY_TYPE(MatrixFill)
BIRCH_UNARY_CALL(MatrixFill, fill, R, C)

}
