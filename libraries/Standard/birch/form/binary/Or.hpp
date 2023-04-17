/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::operator||;
using numbirch::or_grad1;
using numbirch::or_grad2;

template<class Left, class Right>
struct Or : public Binary<Left,Right> {
  template<class T, class U>
  Or(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(operator||)
  BIRCH_BINARY_GRAD(or_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
Or<Left,Right> operator||(const Left& l, const Right& r) {
  return Or<Left,Right>(l, r);
}

}
