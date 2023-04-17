/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::chol;
using numbirch::chol_grad;

template<class Middle>
struct Chol : public Unary<Middle> {
  template<class T>
  Chol(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(chol)
  BIRCH_UNARY_GRAD(chol_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Chol<Middle> chol(const Middle& m) {
  return Chol<Middle>(m);
}

}
