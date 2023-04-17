/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::element;
using numbirch::element_grad1;
using numbirch::element_grad2;

template<class Left, class Right>
struct VectorElement : public Binary<Left,Right> {
  template<class T, class U>
  VectorElement(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(element)
  BIRCH_BINARY_GRAD(element_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
VectorElement<Left,Right> element(const Left& l, const Right& r) {
  return VectorElement<Left,Right>(l, r);
}

}
