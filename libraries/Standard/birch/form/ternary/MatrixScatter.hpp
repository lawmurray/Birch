/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<argument Left, argument Middle, argument Right>
struct MatrixScatter {
  BIRCH_TERNARY_FORM(MatrixScatter, R, C)
};

BIRCH_TERNARY(MatrixScatter, scatter, R, C)
BIRCH_TERNARY_GRAD(MatrixScatter, scatter_grad, R, C)

template<argument Left, argument Middle, argument Right>
int rows(const MatrixScatter<Left,Middle,Right>& o) {
  return o.R;
}

template<argument Left, argument Middle, argument Right>
int columns(const MatrixScatter<Left,Middle,Right>& o) {
  return o.C;
}

template<argument Left, argument Middle, argument Right>
int length(const MatrixScatter<Left,Middle,Right>& o) {
  return o.R;
}

template<argument Left, argument Middle, argument Right>
int size(const MatrixScatter<Left,Middle,Right>& o) {
  return o.R*o.C;
}

}
