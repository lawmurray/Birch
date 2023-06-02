/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<class Left, class Middle, class Right>
struct MatrixSingle {
  BIRCH_TERNARY_FORM(MatrixSingle, R, C)
};

BIRCH_TERNARY(MatrixSingle, numbirch::single, R, C)
BIRCH_TERNARY_GRAD(MatrixSingle, numbirch::single_grad, R, C)

template<class Left, class Middle, class Right>
int rows(const MatrixSingle<Left,Middle,Right>& o) {
  return o.R;
}

template<class Left, class Middle, class Right>
int columns(const MatrixSingle<Left,Middle,Right>& o) {
  return o.C;
}

template<class Left, class Middle, class Right>
int length(const MatrixSingle<Left,Middle,Right>& o) {
  return o.R;
}

template<class Left, class Middle, class Right>
int size(const MatrixSingle<Left,Middle,Right>& o) {
  return o.R*o.C;
}

template<class Left, class Middle, class Right>
auto single(const Left& l, const Middle& m, const Right& r, const int R, const int C) {
  return BIRCH_TERNARY_CONSTRUCT(MatrixSingle, R, C);
}

}
