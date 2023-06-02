/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct ScalarDiagonal {
  BIRCH_UNARY_FORM(ScalarDiagonal, n)
};

BIRCH_UNARY(ScalarDiagonal, numbirch::diagonal, n)
BIRCH_UNARY_GRAD(ScalarDiagonal, numbirch::diagonal_grad, n)

template<class Middle>
int rows(const ScalarDiagonal<Middle>& o) {
  return o.n;
}

template<class Middle>
int columns(const ScalarDiagonal<Middle>& o) {
  return o.n;
}

template<class Middle>
int length(const ScalarDiagonal<Middle>& o) {
  return o.n;
}

template<class Middle>
int size(const ScalarDiagonal<Middle>& o) {
  return pow(o.n, 2);
}

template<class Middle>
auto diagonal(const Middle& m, const int n) {
  return BIRCH_UNARY_CONSTRUCT(ScalarDiagonal, n);
}

}
