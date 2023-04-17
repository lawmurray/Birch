/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::stack;
using numbirch::stack_grad1;
using numbirch::stack_grad2;

template<class Left, class Right>
struct MatrixStack : public Binary<Left,Right> {
  template<class T, class U>
  MatrixStack(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(stack)
  BIRCH_BINARY_GRAD(stack_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
MatrixStack<Left,Right> stack(const Left& l, const Right& r) {
  return MatrixStack<Left,Right>(l, r);
}

}
