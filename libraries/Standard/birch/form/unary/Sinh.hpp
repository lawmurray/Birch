/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::sinh;
using numbirch::sinh_grad;

template<class Middle>
struct Sinh : public Unary<Middle> {
  template<class T>
  Sinh(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(sinh)
  BIRCH_UNARY_GRAD(sinh_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Sinh<Middle> sinh(const Middle& m) {
  return Sinh<Middle>(m);
}

}
