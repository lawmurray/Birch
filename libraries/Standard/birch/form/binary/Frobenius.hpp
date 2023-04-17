/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::frobenius;
using numbirch::frobenius_grad1;
using numbirch::frobenius_grad2;

template<class Left, class Right>
struct Frobenius : public Binary<Left,Right> {
  template<class T, class U>
  Frobenius(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(frobenius)
  BIRCH_BINARY_GRAD(frobenius_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
Frobenius<Left,Right> frobenius(const Left& l, const Right& r) {
  return Frobenius<Left,Right>(l, r);
}

}
