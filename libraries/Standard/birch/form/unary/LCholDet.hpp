/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::lcholdet;
using numbirch::lcholdet_grad;

template<class Middle>
struct LCholDet : public Unary<Middle> {
  template<class T>
  LCholDet(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(lcholdet)
  BIRCH_UNARY_GRAD(lcholdet_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
LCholDet<Middle> lcholdet(const Middle& m) {
  return LCholDet<Middle>(m);
}

}
