/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {
using numbirch::single;
using numbirch::single_grad1;
using numbirch::single_grad2;
using numbirch::single_grad3;

template<class Left, class Middle, class Right>
struct MatrixSingle : public Ternary<Left,Middle,Right> {
  template<class T, class U, class V>
  MatrixSingle(T&& x, U&& i, V&& j, Integer m, Integer n) :
      Ternary<Left,Middle,Right>(std::forward<T>(x), std::forward<U>(i),
      std::forward<V>(j)),
      m(m),
      n(n) {
    //
  }

  /**
   * Number of rows.
   */
  Integer m;

  /**
   * Number of columns.
   */
  Integer n;

  BIRCH_TERNARY_FORM(single, m, n)
  BIRCH_TERNARY_GRAD(single_grad, m, n)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Middle, class Right, std::enable_if_t<
    is_delay_v<Left,Middle,Right>,int> = 0>
MatrixSingle<Left,Middle,Right> single(const Left& x, const Middle& i,
    const Right& j, const int m, const int n) {
  return MatrixSingle<Left,Middle,Right>(x, i, j, m, n);
}

}
