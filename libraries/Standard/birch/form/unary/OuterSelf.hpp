/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::outer;
using numbirch::outer_grad;

template<class Middle>
struct OuterSelf : public Unary<Middle> {
  template<class T>
  OuterSelf(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(outer)
  BIRCH_UNARY_GRAD(outer_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
OuterSelf<Middle> outer(const Middle& m) {
  return OuterSelf<Middle>(m);
}

}
