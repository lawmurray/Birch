/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::ceil;
using numbirch::ceil_grad;

template<class Middle>
struct Ceil : public Unary<Middle> {
  template<class T>
  Ceil(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(ceil)
  BIRCH_UNARY_GRAD(ceil_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Ceil<Middle> ceil(const Middle& m) {
  return Ceil<Middle>(m);
}

}
