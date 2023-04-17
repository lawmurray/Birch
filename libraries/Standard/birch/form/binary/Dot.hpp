/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::dot;
using numbirch::dot_grad1;
using numbirch::dot_grad2;

template<class Left, class Right>
struct Dot : public Binary<Left,Right> {
  template<class T, class U>
  Dot(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(dot)
  BIRCH_BINARY_GRAD(dot_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
Dot<Left,Right> dot(const Left& l, const Right& r) {
  return Dot<Left,Right>(l, r);
}

}
