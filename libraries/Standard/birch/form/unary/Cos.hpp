/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::cos;
using numbirch::cos_grad;

template<class Middle>
struct Cos : public Unary<Middle> {
  template<class T>
  Cos(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(cos)
  BIRCH_UNARY_GRAD(cos_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Cos<Middle> cos(const Middle& m) {
  return Cos<Middle>(m);
}

}
