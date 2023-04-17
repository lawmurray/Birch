/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::inv;
using numbirch::inv_grad;

template<class Middle>
struct Inv : public Unary<Middle> {
  template<class T>
  Inv(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(inv)
  BIRCH_UNARY_GRAD(inv_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Inv<Middle> inv(const Middle& m) {
  return Inv<Middle>(m);
}

}
