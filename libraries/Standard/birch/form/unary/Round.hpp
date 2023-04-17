/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::round;
using numbirch::round_grad;

template<class Middle>
struct Round : public Unary<Middle> {
  template<class T>
  Round(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(round)
  BIRCH_UNARY_GRAD(round_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Round<Middle> round(const Middle& m) {
  return Round<Middle>(m);
}

}
