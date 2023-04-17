/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::scal;
using numbirch::scal_grad;

template<class Middle>
struct Scal : public Unary<Middle> {
  template<class T>
  Scal(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(scal)
  BIRCH_UNARY_GRAD(scal_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Scal<Middle> scal(const Middle& m) {
  return Scal<Middle>(m);
}

}
