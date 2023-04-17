/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::atan;
using numbirch::atan_grad;

template<class Middle>
struct Atan : public Unary<Middle> {
  template<class T>
  Atan(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(atan)
  BIRCH_UNARY_GRAD(atan_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Atan<Middle> atan(const Middle& m) {
  return Atan<Middle>(m);
}

}
