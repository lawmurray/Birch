/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::single;
using numbirch::single_grad1;
using numbirch::single_grad2;

template<class Left, class Right>
struct VectorSingle : public Binary<Left,Right> {
  /**
   * Length of vector.
   */
  Integer n;

  template<class T, class U>
  VectorSingle(T&& x, U&& i, Integer n) :
      Binary<Left,Right>(std::forward<T>(x), std::forward<U>(i)),
      n(n) {
    //
  }

  BIRCH_BINARY_FORM(single, n)
  BIRCH_BINARY_GRAD(single_grad, n)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
VectorSingle<Left,Right> single(const Left& x, const Right& i, const int n) {
  return VectorSingle<Left,Right>(x, i, n);
}

}
