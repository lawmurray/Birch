/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct ScalarDiagonal {
  BIRCH_UNARY_FORM(ScalarDiagonal, n)
  BIRCH_UNARY_EVAL(ScalarDiagonal, diagonal, n)
  BIRCH_UNARY_GRAD(ScalarDiagonal, diagonal_grad, n)

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

BIRCH_UNARY_TYPE(ScalarDiagonal)
BIRCH_UNARY_CALL(ScalarDiagonal, diagonal, n)

}
