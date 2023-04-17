/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {
using numbirch::scatter;
using numbirch::scatter_grad1;
using numbirch::scatter_grad2;
using numbirch::scatter_grad3;

template<class Left, class Middle, class Right>
struct MatrixScatter : public Ternary<Left,Middle,Right> {
  template<class T, class U, class V>
  MatrixScatter(T&& A, U&& I, V&& J, Integer m, Integer n) :
      Ternary<Left,Middle,Right>(std::forward<T>(A), std::forward<U>(I),
      std::forward<V>(J)),
      k(m),
      l(n) {
    //
  }

  /**
   * Number of rows of result.
   */
  Integer k;

  /**
   * Number of columns of result.
   */
  Integer l;

  BIRCH_TERNARY_FORM(scatter, k, l)
  BIRCH_TERNARY_GRAD(scatter_grad, k, l)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Middle, class Right, std::enable_if_t<
    is_delay_v<Left,Middle,Right>,int> = 0>
MatrixScatter<Left,Middle,Right> scatter(const Left& A, const Middle& I,
    const Right& J, const int m, const int n) {
  return MatrixScatter<Left,Middle,Right>(A, I, J, m, n);
}

}
