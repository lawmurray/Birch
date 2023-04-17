/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::pow;
using numbirch::pow_grad1;
using numbirch::pow_grad2;

template<class Left, class Right>
struct Pow : public Binary<Left,Right> {
  template<class T, class U>
  Pow(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(pow)
  BIRCH_BINARY_GRAD(pow_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
Pow<Left,Right> pow(const Left& l, const Right& r) {
  return Pow<Left,Right>(l, r);
}

}
