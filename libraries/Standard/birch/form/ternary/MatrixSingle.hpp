/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<argument Left, argument Middle, argument Right>
struct MatrixSingle {
  BIRCH_TERNARY_FORM(MatrixSingle, R, C)
};

BIRCH_TERNARY(MatrixSingle, single, R, C)
BIRCH_TERNARY_GRAD(MatrixSingle, single_grad, R, C)

template<argument Left, argument Middle, argument Right>
int rows(const MatrixSingle<Left,Middle,Right>& o) {
  return o.R;
}

template<argument Left, argument Middle, argument Right>
int columns(const MatrixSingle<Left,Middle,Right>& o) {
  return o.C;
}

template<argument Left, argument Middle, argument Right>
int length(const MatrixSingle<Left,Middle,Right>& o) {
  return o.R;
}

template<argument Left, argument Middle, argument Right>
int size(const MatrixSingle<Left,Middle,Right>& o) {
  return o.R*o.C;
}

}
