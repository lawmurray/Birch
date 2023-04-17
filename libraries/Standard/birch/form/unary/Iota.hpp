/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::iota;
using numbirch::iota_grad;

template<class Middle>
struct Iota : public Unary<Middle> {
  /**
   * Length.
   */
  Integer n;

  template<class T>
  Iota(T&& m, Integer n) :
      Unary<Middle>(std::forward<T>(m)),
      n(n) {
    //
  }

  BIRCH_UNARY_FORM(iota, n)
  BIRCH_UNARY_GRAD(iota_grad, n)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Iota<Middle> iota(const Middle& m, const int n) {
  return Iota<Middle>(m, n);
}

}
