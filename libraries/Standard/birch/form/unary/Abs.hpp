/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::abs;
using numbirch::abs_grad;

template<class Middle>
struct Abs : public Unary<Middle> {
  template<class T>
  Abs(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(abs)
  BIRCH_UNARY_GRAD(abs_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Abs<Middle> abs(const Middle& m) {
  return Abs<Middle>(m);
}

}
