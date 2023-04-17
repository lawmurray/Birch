/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::diagonal;
using numbirch::diagonal_grad;

template<class Middle>
struct DiagonalScalar : public Unary<Middle> {
  /**
   * Number of rows and columns.
   */
  Integer n;

  template<class T>
  DiagonalScalar(T&& m, Integer n) :
      Unary<Middle>(std::forward<T>(m)),
      n(n) {
    //
  }

  BIRCH_FORM_OP
  BIRCH_UNARY_FORM(diagonal, n)
  BIRCH_UNARY_GRAD(diagonal_grad, n)

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
struct DiagonalVector : public Unary<Middle> {
  template<class T>
  DiagonalVector(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_FORM_OP
  BIRCH_UNARY_FORM(diagonal)
  BIRCH_UNARY_GRAD(diagonal_grad)

  int rows() const {
    return length(x);
  }

  int columns() const {
    return length(x);
  }

  int length() const {
    return length(x);
  }

  int size() const {
    return pow(length(x), 2);
  }
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
DiagonalScalar<Middle> diagonal(const Middle& m, const int n) {
  return DiagonalScalar<Middle>(m, n);
}

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
DiagonalVector<Middle> diagonal(const Middle& m) {
  return DiagonalVector<Middle>(m);
}

}
