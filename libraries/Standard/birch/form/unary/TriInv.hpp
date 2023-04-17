/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::triinv;
using numbirch::triinv_grad;

template<class Middle>
struct TriInv : public Unary<Middle> {
  template<class T>
  TriInv(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(triinv)
  BIRCH_UNARY_GRAD(triinv_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
TriInv<Middle> triinv(const Middle& m) {
  return TriInv<Middle>(m);
}

}
