/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::isnan;
using numbirch::isnan_grad;

template<class Middle>
struct IsNan : public Unary<Middle> {
  template<class T>
  IsNan(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(isnan)
  BIRCH_UNARY_GRAD(isnan_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
IsNan<Middle> isnan(const Middle& m) {
  return IsNan<Middle>(m);
}

}
