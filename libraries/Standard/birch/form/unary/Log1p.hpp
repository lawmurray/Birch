/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::log1p;
using numbirch::log1p_grad;

template<class Middle>
struct Log1p : public Unary<Middle> {
  template<class T>
  Log1p(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(log1p)
  BIRCH_UNARY_GRAD(log1p_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Log1p<Middle> log1p(const Middle& m) {
  return Log1p<Middle>(m);
}

}
