/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::trisolve;
using numbirch::trisolve_grad1;
using numbirch::trisolve_grad2;

template<class Left, class Right>
struct TriSolve : public Binary<Left,Right> {
  template<class T, class U>
  TriSolve(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(trisolve)
  BIRCH_BINARY_GRAD(trisolve_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
TriSolve<Left,Right> trisolve(const Left& l, const Right& r) {
  return TriSolve<Left,Right>(l, r);
}

}
