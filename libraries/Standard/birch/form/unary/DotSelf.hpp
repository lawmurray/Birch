/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::dot;
using numbirch::dot_grad;

template<class Middle>
struct DotSelf : public Unary<Middle> {
  template<class T>
  DotSelf(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(dot)
  BIRCH_UNARY_GRAD(dot_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
DotSelf<Middle> dot(const Middle& m) {
  return DotSelf<Middle>(m);
}

}
