/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::copysign;
using numbirch::copysign_grad1;
using numbirch::copysign_grad2;

template<class Left, class Right>
struct CopySign : public Binary<Left,Right> {
  template<class T, class U>
  CopySign(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(copysign)
  BIRCH_BINARY_GRAD(copysign_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
CopySign<Left,Right> copysign(const Left& l, const Right& r) {
  return CopySign<Left,Right>(l, r);
}

}
