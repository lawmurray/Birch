/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::outer;
using numbirch::outer_grad1;
using numbirch::outer_grad2;

template<class Left, class Right>
struct Outer : public Binary<Left,Right> {
  template<class T, class U>
  Outer(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(outer)
  BIRCH_BINARY_GRAD(outer_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
Outer<Left,Right> outer(const Left& l, const Right& r) {
  return Outer<Left,Right>(l, r);
}

}
