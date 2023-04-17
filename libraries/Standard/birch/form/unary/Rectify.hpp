/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::rectify;
using numbirch::rectify_grad;

template<class Middle>
struct Rectify : public Unary<Middle> {
  template<class T>
  Rectify(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(rectify)
  BIRCH_UNARY_GRAD(rectify_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Rectify<Middle> rectify(const Middle& m) {
  return Rectify<Middle>(m);
}

}
