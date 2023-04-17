/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::inner;
using numbirch::inner_grad;

template<class Middle>
struct InnerSelf : public Unary<Middle> {
  template<class T>
  InnerSelf(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(inner)
  BIRCH_UNARY_GRAD(inner_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
InnerSelf<Middle> inner(const Middle& m) {
  return InnerSelf<Middle>(m);
}

}
