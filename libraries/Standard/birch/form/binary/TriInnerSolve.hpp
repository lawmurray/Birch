/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::triinnersolve;
using numbirch::triinnersolve_grad1;
using numbirch::triinnersolve_grad2;

template<class Left, class Right>
struct TriInnerSolve : public Binary<Left,Right> {
  template<class T, class U>
  TriInnerSolve(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(triinnersolve)
  BIRCH_BINARY_GRAD(triinnersolve_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
TriInnerSolve<Left,Right> triinnersolve(const Left& l, const Right& r) {
  return TriInnerSolve<Left,Right>(l, r);
}

}
