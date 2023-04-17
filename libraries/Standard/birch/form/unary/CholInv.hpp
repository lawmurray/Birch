/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::cholinv;
using numbirch::cholinv_grad;

template<class Middle>
struct CholInv : public Unary<Middle> {
  template<class T>
  CholInv(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(cholinv)
  BIRCH_UNARY_GRAD(cholinv_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
CholInv<Middle> cholinv(const Middle& m) {
  return CholInv<Middle>(m);
}

}
