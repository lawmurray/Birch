/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::scatter;
using numbirch::scatter_grad1;
using numbirch::scatter_grad2;

template<class Left, class Right>
struct VectorScatter : public Binary<Left,Right> {
  /**
   * Length of result.
   */
  Integer n;

  template<class T, class U>
  VectorScatter(T&& l, U&& r, Integer n) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)),
      n(n) {
    //
  }

  BIRCH_BINARY_FORM(scatter, n)
  BIRCH_BINARY_GRAD(scatter_grad, n)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
VectorScatter<Left,Right> scatter(const Left& l, const Right& r,
    const int n) {
  return VectorScatter<Left,Right>(l, r, n);
}

}
