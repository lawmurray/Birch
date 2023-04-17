/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::asin;
using numbirch::asin_grad;

template<class Middle>
struct Asin : public Unary<Middle> {
  template<class T>
  Asin(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(asin)
  BIRCH_UNARY_GRAD(asin_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Asin<Middle> asin(const Middle& m) {
  return Asin<Middle>(m);
}

}
