/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::fill;
using numbirch::fill_grad;

template<class Middle>
struct VectorFill : public Unary<Middle> {
  /**
   * Length.
   */
  Integer n;

  template<class T>
  VectorFill(T&& m, Integer n) :
      Unary<Middle>(std::forward<T>(m)),
      n(n) {
    //
  }

  BIRCH_FORM_OP
  BIRCH_UNARY_FORM(fill, n)
  BIRCH_UNARY_GRAD(fill_grad, n)

  int rows() const {
    return n;
  }

  int columns() const {
    return n;
  }

  int length() const {
    return n;
  }

  int size() const {
    return n;
  }
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
VectorFill<Middle> fill(const Middle& m, const int n) {
  return VectorFill<Middle>(m, n);
}

}
