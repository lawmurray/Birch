/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct ScalarDiagonal {
  BIRCH_UNARY_FORM(ScalarDiagonal, numbirch::diagonal, n)
  BIRCH_UNARY_GRAD(numbirch::diagonal_grad, n)

  int rows() const {
    return n;
  }

  int columns() const {
    return n;
  }

  int length() const {
    return n;
  }

  int size() const {
    return n*n;
  }
};

template<class Middle>
auto diagonal(const Middle& m, const int n) {
  return BIRCH_UNARY_CONSTRUCT(ScalarDiagonal, n);
}

}
