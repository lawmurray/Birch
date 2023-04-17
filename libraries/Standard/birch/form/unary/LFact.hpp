/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::lfact;
using numbirch::lfact_grad;

template<class Middle>
struct LFact : public Unary<Middle> {
  template<class T>
  LFact(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(lfact)
  BIRCH_UNARY_GRAD(lfact_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
LFact<Middle> lfact(const Middle& m) {
  return LFact<Middle>(m);
}

}
