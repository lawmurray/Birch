/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::ltridet;
using numbirch::ltridet_grad;

template<class Middle>
struct LTriDet : public Unary<Middle> {
  template<class T>
  LTriDet(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(ltridet)
  BIRCH_UNARY_GRAD(ltridet_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
LTriDet<Middle> ltridet(const Middle& m) {
  return LTriDet<Middle>(m);
}

}
