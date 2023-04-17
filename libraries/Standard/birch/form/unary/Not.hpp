/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::operator!;
using numbirch::not_grad;

template<class Middle>
struct Not : public Unary<Middle> {
  template<class T>
  Not(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(operator!)
  BIRCH_UNARY_GRAD(not_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Not<Middle> operator!(const Middle& m) {
  return Not<Middle>(m);
}

}
