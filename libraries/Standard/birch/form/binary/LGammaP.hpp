/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::lgamma;
using numbirch::lgamma_grad1;
using numbirch::lgamma_grad2;

template<class Left, class Right>
struct LGammaP : public Binary<Left,Right> {
  template<class T, class U>
  LGammaP(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(lgamma)
  BIRCH_BINARY_GRAD(lgamma_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
LGammaP<Left,Right> lgamma(const Left& l, const Right& r) {
  return LGammaP<Left,Right>(l, r);
}

}
