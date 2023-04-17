/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::frobenius;
using numbirch::frobenius_grad;

template<class Middle>
struct FrobeniusSelf : public Unary<Middle> {
  template<class T>
  FrobeniusSelf(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(frobenius)
  BIRCH_UNARY_GRAD(frobenius_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
FrobeniusSelf<Middle> frobenius(const Middle& m) {
  return FrobeniusSelf<Middle>(m);
}

}
