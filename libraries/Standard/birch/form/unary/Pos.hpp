/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::operator+;
using numbirch::pos_grad;

template<class Middle>
struct Pos : public Unary<Middle> {
  template<class T>
  Pos(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(operator+)
  BIRCH_UNARY_GRAD(pos_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Pos<Middle> operator+(const Middle& m) {
  return Pos<Middle>(m);
}

}
