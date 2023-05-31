/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<class Left, class Middle, class Right>
struct MatrixScatter {
  BIRCH_TERNARY_FORM(MatrixScatter, numbirch::scatter, R, C)
  BIRCH_TERNARY_GRAD(numbirch::scatter_grad, R, C)
  BIRCH_FORM
};

template<class Left, class Middle, class Right>
auto scatter(const Left& l, const Middle& m, const Right& r, const int R, const int C) {
  return BIRCH_TERNARY_CONSTRUCT(MatrixScatter, R, C);
}

}
