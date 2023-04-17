/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::expm1;
using numbirch::expm1_grad;

template<class Middle>
struct Expm1 : public Unary<Middle> {
  template<class T>
  Expm1(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(expm1)
  BIRCH_UNARY_GRAD(expm1_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Expm1<Middle> expm1(const Middle& m) {
  return Expm1<Middle>(m);
}

}
