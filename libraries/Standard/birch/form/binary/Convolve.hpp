/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::convolve;
using numbirch::convolve_grad1;
using numbirch::convolve_grad2;

template<class Left, class Right>
struct Convolve : public Binary<Left,Right> {
  template<class T, class U>
  Convolve(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(convolve)
  BIRCH_BINARY_GRAD(convolve_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
Convolve<Left,Right> convolve(const Left& l, const Right& r) {
  return Convolve<Left,Right>(l, r);
}

}
