/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::triinner;
using numbirch::triinner_grad;

template<class Middle>
struct TriInnerSelf : public Unary<Middle> {
  template<class T>
  TriInnerSelf(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(triinner)
  BIRCH_UNARY_GRAD(triinner_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
TriInnerSelf<Middle> triinner(const Middle& m) {
  return TriInnerSelf<Middle>(m);
}

}
