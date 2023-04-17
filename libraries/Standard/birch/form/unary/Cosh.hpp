/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::cosh;
using numbirch::cosh_grad;

template<class Middle>
struct Cosh : public Unary<Middle> {
  template<class T>
  Cosh(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(cosh)
  BIRCH_UNARY_GRAD(cosh_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Cosh<Middle> cosh(const Middle& m) {
  return Cosh<Middle>(m);
}

}
