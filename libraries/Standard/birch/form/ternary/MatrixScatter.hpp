/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<class Left, class Middle, class Right>
struct MatrixScatter {
  BIRCH_TERNARY_FORM(MatrixScatter, R, C)
};

BIRCH_TERNARY(MatrixScatter, numbirch::scatter, R, C)
BIRCH_TERNARY_GRAD(MatrixScatter, numbirch::scatter_grad, R, C)

template<class Left, class Middle, class Right>
int rows(const MatrixScatter<Left,Middle,Right>& o) {
  return o.R;
}

template<class Left, class Middle, class Right>
int columns(const MatrixScatter<Left,Middle,Right>& o) {
  return o.C;
}

template<class Left, class Middle, class Right>
int length(const MatrixScatter<Left,Middle,Right>& o) {
  return o.R;
}

template<class Left, class Middle, class Right>
int size(const MatrixScatter<Left,Middle,Right>& o) {
  return o.R*o.C;
}

template<class Left, class Middle, class Right>
auto scatter(const Left& l, const Middle& m, const Right& r, const int R, const int C) {
  return BIRCH_TERNARY_CONSTRUCT(MatrixScatter, R, C);
}

}
