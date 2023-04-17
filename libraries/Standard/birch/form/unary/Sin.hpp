/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::sin;
using numbirch::sin_grad;

template<class Middle>
struct Sin : public Unary<Middle> {
  template<class T>
  Sin(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(sin)
  BIRCH_UNARY_GRAD(sin_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Sin<Middle> sin(const Middle& m) {
  return Sin<Middle>(m);
}

}
