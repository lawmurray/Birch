/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::ldet;
using numbirch::ldet_grad;

template<class Middle>
struct LDet : public Unary<Middle> {
  template<class T>
  LDet(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(ldet)
  BIRCH_UNARY_GRAD(ldet_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
LDet<Middle> ldet(const Middle& m) {
  return LDet<Middle>(m);
}

}
