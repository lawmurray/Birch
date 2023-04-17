/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::cholsolve;
using numbirch::cholsolve_grad1;
using numbirch::cholsolve_grad2;

template<class Left, class Right>
struct CholSolve : public Binary<Left,Right> {
  template<class T, class U>
  CholSolve(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(cholsolve)
  BIRCH_BINARY_GRAD(cholsolve_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
CholSolve<Left,Right> cholsolve(const Left& l, const Right& r) {
  return CholSolve<Left,Right>(l, r);
}

}
