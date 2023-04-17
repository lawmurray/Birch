/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::isfinite;
using numbirch::isfinite_grad;

template<class Middle>
struct IsFinite : public Unary<Middle> {
  template<class T>
  IsFinite(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(isfinite)
  BIRCH_UNARY_GRAD(isfinite_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
IsFinite<Middle> isfinite(const Middle& m) {
  return IsFinite<Middle>(m);
}

}
