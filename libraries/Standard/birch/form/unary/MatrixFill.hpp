/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::fill;
using numbirch::fill_grad;

template<class Middle>
struct MatrixFill : public Unary<Middle> {
  /**
   * Number of rows.
   */
  Integer k;

  /**
   * Number of columns.
   */
  Integer l;

  template<class T>
  MatrixFill(T&& a, Integer m, Integer n) :
      Unary<Middle>(std::forward<T>(a)),
      k(m),
      l(n) {
    //
  }

  BIRCH_FORM_OP
  BIRCH_UNARY_FORM(fill, k, l)
  BIRCH_UNARY_GRAD(fill_grad, k, l)

  int rows() const {
    return k;
  }

  int columns() const {
    return l;
  }

  int length() const {
    return k;
  }

  int size() const {
    return k*l;
  }
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
MatrixFill<Middle> fill(const Middle& a, const int m, const int n) {
  return MatrixFill<Middle>(a, m, n);
}

}
