/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<class Left, class Middle, class Right>
struct MatrixSingle {
  BIRCH_TERNARY_FORM(MatrixSingle, numbirch::single, R, C)
  BIRCH_TERNARY_GRAD(numbirch::single_grad, R, C)
  BIRCH_FORM
};

template<class Left, class Middle, class Right>
auto single(const Left& l, const Middle& m, const Right& r, const int R, const int C) {
  return BIRCH_TERNARY_CONSTRUCT(MatrixSingle, R, C);
}

}
