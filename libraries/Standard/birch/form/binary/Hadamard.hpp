/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::hadamard;
using numbirch::hadamard_grad1;
using numbirch::hadamard_grad2;

template<class Left, class Right>
struct Hadamard : public Binary<Left,Right> {
  template<class T, class U>
  Hadamard(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(hadamard)
  BIRCH_BINARY_GRAD(hadamard_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
Hadamard<Left,Right> hadamard(const Left& l, const Right& r) {
  return Hadamard<Left,Right>(l, r);
}

}
