/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::lbeta;
using numbirch::lbeta_grad1;
using numbirch::lbeta_grad2;

template<class Left, class Right>
struct LBeta : public Binary<Left,Right> {
  template<class T, class U>
  LBeta(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(lbeta)
  BIRCH_BINARY_GRAD(lbeta_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
LBeta<Left,Right> lbeta(const Left& l, const Right& r) {
  return LBeta<Left,Right>(l, r);
}

}
