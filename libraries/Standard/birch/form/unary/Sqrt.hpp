/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::sqrt;
using numbirch::sqrt_grad;

template<class Middle>
struct Sqrt : public Unary<Middle> {
  template<class T>
  Sqrt(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(sqrt)
  BIRCH_UNARY_GRAD(sqrt_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Sqrt<Middle> sqrt(const Middle& m) {
  return Sqrt<Middle>(m);
}

}
