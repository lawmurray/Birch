/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::exp;
using numbirch::exp_grad;

template<class Middle>
struct Exp : public Unary<Middle> {
  template<class T>
  Exp(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(exp)
  BIRCH_UNARY_GRAD(exp_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Exp<Middle> exp(const Middle& m) {
  return Exp<Middle>(m);
}

}
