/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::operator==;
using numbirch::equal_grad1;
using numbirch::equal_grad2;

template<class Left, class Right>
struct Equal : public Binary<Left,Right> {
  template<class T, class U>
  Equal(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(operator==)
  BIRCH_BINARY_GRAD(equal_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
Equal<Left,Right> operator==(const Left& l, const Right& r) {
  return Equal<Left,Right>(l, r);
}

}
