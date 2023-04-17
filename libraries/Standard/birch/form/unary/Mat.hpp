/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::mat;
using numbirch::mat_grad;

template<class Middle>
struct Mat : public Unary<Middle> {
  /**
   * Number of columns.
   */
  Integer n;

  template<class T>
  Mat(T&& m, Integer n) :
      Unary<Middle>(std::forward<T>(m)),
      n(n) {
    //
  }

  BIRCH_UNARY_FORM(mat, n)
  BIRCH_UNARY_GRAD(mat_grad, n)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Mat<Middle> mat(const Middle& m, const int n) {
  return Mat<Middle>(m, n);
}

}
