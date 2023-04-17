/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::triinner;
using numbirch::triinner_grad1;
using numbirch::triinner_grad2;

template<class Left, class Right>
struct TriInner : public Binary<Left,Right> {
  template<class T, class U>
  TriInner(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(triinner)
  BIRCH_BINARY_GRAD(triinner_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
TriInner<Left,Right> triinner(const Left& l, const Right& r) {
  return TriInner<Left,Right>(l, r);
}

}
