/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct VectorDiagonal {
  BIRCH_UNARY_FORM(VectorDiagonal, numbirch::diagonal)
  BIRCH_UNARY_GRAD(numbirch::diagonal_grad)

  int rows() const {
    return length(peek());
  }

  int columns() const {
    return length(peek());
  }

  int length() const {
    return length(peek());
  }

  int size() const {
    return pow(length(peek()), 2);
  }
};

template<class Middle>
auto diagonal(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(VectorDiagonal);
}

}
