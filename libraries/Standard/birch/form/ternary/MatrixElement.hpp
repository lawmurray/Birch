/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {
using numbirch::element;
using numbirch::element_grad1;
using numbirch::element_grad2;
using numbirch::element_grad3;

template<class Left, class Middle, class Right>
struct MatrixElement : public Ternary<Left,Middle,Right> {
  template<class T, class U, class V>
  MatrixElement(T&& l, U&& m, V&& r) :
      Ternary<Left,Middle,Right>(std::forward<T>(l), std::forward<U>(m),
      std::forward<V>(r)) {
    //
  }

  BIRCH_TERNARY_FORM(element)
  BIRCH_TERNARY_GRAD(element_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Middle, class Right, std::enable_if_t<
    is_delay_v<Left,Middle,Right>,int> = 0>
MatrixElement<Left,Middle,Right> element(const Left& l, const Middle& m,
    const Right& r) {
  return MatrixElement<Left,Middle,Right>(l, m, r);
}

}
