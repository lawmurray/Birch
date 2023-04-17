/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::operator-;
using numbirch::neg_grad;

template<class Middle>
struct Neg : public Unary<Middle> {
  template<class T>
  Neg(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(operator-)
  BIRCH_UNARY_GRAD(neg_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Neg<Middle> operator-(const Middle& m) {
  return Neg<Middle>(m);
}

}
