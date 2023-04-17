/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::transpose;
using numbirch::transpose_grad;

template<class Middle>
struct Transpose : public Unary<Middle> {
  template<class T>
  Transpose(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(transpose)
  BIRCH_UNARY_GRAD(transpose_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Transpose<Middle> transpose(const Middle& m) {
  return Transpose<Middle>(m);
}

}
