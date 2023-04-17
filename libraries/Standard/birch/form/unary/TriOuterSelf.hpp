/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::triouter;
using numbirch::triouter_grad;

template<class Middle>
struct TriOuterSelf : public Unary<Middle> {
  template<class T>
  TriOuterSelf(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(triouter)
  BIRCH_UNARY_GRAD(triouter_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
TriOuterSelf<Middle> triouter(const Middle& m) {
  return TriOuterSelf<Middle>(m);
}

}
