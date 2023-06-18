/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct VectorDiagonal {
  BIRCH_UNARY_FORM(VectorDiagonal)
  BIRCH_UNARY_EVAL(VectorDiagonal, diagonal)
  BIRCH_UNARY_GRAD(VectorDiagonal, diagonal_grad)

  int rows() const {
    return length(m);
  }

  int columns() const {
    return length(m);
  }

  int length() const {
    return length(m);
  }

  int size() const {
    return pow(length(m), 2);
  }
};

BIRCH_UNARY_TYPE(VectorDiagonal)
BIRCH_UNARY_CALL(VectorDiagonal, diagonal)

}
