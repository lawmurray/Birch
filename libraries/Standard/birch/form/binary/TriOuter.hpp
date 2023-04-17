/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::triouter;
using numbirch::triouter_grad1;
using numbirch::triouter_grad2;

template<class Left, class Right>
struct TriOuter : public Binary<Left,Right> {
  template<class T, class U>
  TriOuter(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(triouter)
  BIRCH_BINARY_GRAD(triouter_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
TriOuter<Left,Right> triouter(const Left& l, const Right& r) {
  return TriOuter<Left,Right>(l, r);
}

}
