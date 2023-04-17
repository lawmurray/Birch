/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::div;
using numbirch::div_grad1;
using numbirch::div_grad2;

template<class Left, class Right>
struct Div : public Binary<Left,Right> {
  template<class T, class U>
  Div(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(div)
  BIRCH_BINARY_GRAD(div_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
Div<Left,Right> div(const Left& l, const Right& r) {
  return Div<Left,Right>(l, r);
}

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
Div<Left,Right> operator/(const Left& l, const Right& r) {
  return Div<Left,Right>(l, r);
}

}
