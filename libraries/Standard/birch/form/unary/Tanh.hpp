/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::tanh;
using numbirch::tanh_grad;

template<class Middle>
struct Tanh : public Unary<Middle> {
  template<class T>
  Tanh(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(tanh)
  BIRCH_UNARY_GRAD(tanh_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Tanh<Middle> tanh(const Middle& m) {
  return Tanh<Middle>(m);
}

}
