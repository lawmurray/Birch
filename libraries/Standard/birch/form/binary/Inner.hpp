/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::inner;
using numbirch::inner_grad1;
using numbirch::inner_grad2;

template<class Left, class Right>
struct Inner : public Binary<Left,Right> {
  template<class T, class U>
  Inner(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(inner)
  BIRCH_BINARY_GRAD(inner_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
Inner<Left,Right> inner(const Left& l, const Right& r) {
  return Inner<Left,Right>(l, r);
}

}
