/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::lgamma;
using numbirch::lgamma_grad;

template<class Middle>
struct LGamma : public Unary<Middle> {
  template<class T>
  LGamma(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(lgamma)
  BIRCH_UNARY_GRAD(lgamma_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
LGamma<Middle> lgamma(const Middle& m) {
  return LGamma<Middle>(m);
}

}
