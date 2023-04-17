/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::isinf;
using numbirch::isinf_grad;

template<class Middle>
struct IsInf : public Unary<Middle> {
  template<class T>
  IsInf(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(isinf)
  BIRCH_UNARY_GRAD(isinf_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
IsInf<Middle> isinf(const Middle& m) {
  return IsInf<Middle>(m);
}

}
