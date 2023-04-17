/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::lchoose;
using numbirch::lchoose_grad1;
using numbirch::lchoose_grad2;

template<class Left, class Right>
struct LChoose : public Binary<Left,Right> {
  template<class T, class U>
  LChoose(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(lchoose)
  BIRCH_BINARY_GRAD(lchoose_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
LChoose<Left,Right> lchoose(const Left& l, const Right& r) {
  return LChoose<Left,Right>(l, r);
}

}
