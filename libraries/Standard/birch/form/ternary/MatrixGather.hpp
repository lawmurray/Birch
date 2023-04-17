/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {
using numbirch::gather;
using numbirch::gather_grad1;
using numbirch::gather_grad2;
using numbirch::gather_grad3;

template<class Left, class Middle, class Right>
struct MatrixGather : public Ternary<Left,Middle,Right> {
  template<class T, class U, class V>
  MatrixGather(T&& A, U&& I, V&& J) :
      Ternary<Left,Middle,Right>(std::forward<T>(A), std::forward<U>(I),
      std::forward<V>(J)) {
    //
  }

  BIRCH_TERNARY_FORM(gather)
  BIRCH_TERNARY_GRAD(gather_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Middle, class Right, std::enable_if_t<
    is_delay_v<Left,Middle,Right>,int> = 0>
MatrixGather<Left,Middle,Right> gather(const Left& l, const Middle& m,
    const Right& r) {
  return MatrixGather<Left,Middle,Right>(l, m, r);
}

}
