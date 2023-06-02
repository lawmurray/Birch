/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct MatrixFill {
  BIRCH_UNARY_FORM(MatrixFill, R, C)
};

BIRCH_UNARY(MatrixFill, numbirch::fill, R, C)
BIRCH_UNARY_GRAD(MatrixFill, numbirch::fill_grad, R, C)

template<class Middle>
int rows(const MatrixFill<Middle>& o) {
  return o.R;
}

template<class Middle>
int columns(const MatrixFill<Middle>& o) {
  return o.C;
}

template<class Middle>
int length(const MatrixFill<Middle>& o) {
  return o.R;
}

template<class Middle>
int size(const MatrixFill<Middle>& o) {
  return o.R*o.C;
}

template<class Middle>
auto fill(const Middle& m, const int R, const int C) {
  return BIRCH_UNARY_CONSTRUCT(MatrixFill, R, C);
}

}
