/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct MatrixFill {
  BIRCH_UNARY_FORM(MatrixFill, numbirch::fill, R, C)
  BIRCH_UNARY_GRAD(numbirch::fill_grad, R, C)

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

template<class Middle>
auto fill(const Middle& m, const int R, const int C) {
  return BIRCH_UNARY_CONSTRUCT(MatrixFill, R, C);
}

}
