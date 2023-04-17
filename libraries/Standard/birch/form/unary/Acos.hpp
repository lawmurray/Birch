/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::acos;
using numbirch::acos_grad;

template<class Middle>
struct Acos : public Unary<Middle> {
  template<class T>
  Acos(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(acos)
  BIRCH_UNARY_GRAD(acos_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Acos<Middle> acos(const Middle& m) {
  return Acos<Middle>(m);
}

}
