/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::tan;
using numbirch::tan_grad;

template<class Middle>
struct Tan : public Unary<Middle> {
  template<class T>
  Tan(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(tan)
  BIRCH_UNARY_GRAD(tan_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Tan<Middle> tan(const Middle& m) {
  return Tan<Middle>(m);
}

}
