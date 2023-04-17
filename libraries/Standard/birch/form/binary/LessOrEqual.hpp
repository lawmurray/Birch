/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::operator<=;
using numbirch::less_or_equal_grad1;
using numbirch::less_or_equal_grad2;

template<class Left, class Right>
struct LessOrEqual : public Binary<Left,Right> {
  template<class T, class U>
  LessOrEqual(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(operator<=)
  BIRCH_BINARY_GRAD(less_or_equal_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
LessOrEqual<Left,Right> operator<=(const Left& l, const Right& r) {
  return LessOrEqual<Left,Right>(l, r);
}

}
