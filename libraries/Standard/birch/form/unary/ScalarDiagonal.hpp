/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct ScalarDiagonal {
  BIRCH_UNARY_FORM(ScalarDiagonal, n)
};

BIRCH_UNARY(ScalarDiagonal, diagonal, n)
BIRCH_UNARY_GRAD(ScalarDiagonal, diagonal_grad, n)

template<argument Middle>
int rows(const ScalarDiagonal<Middle>& o) {
  return o.n;
}

template<argument Middle>
int columns(const ScalarDiagonal<Middle>& o) {
  return o.n;
}

template<argument Middle>
int length(const ScalarDiagonal<Middle>& o) {
  return o.n;
}

template<argument Middle>
int size(const ScalarDiagonal<Middle>& o) {
  return pow(o.n, 2);
}

}
