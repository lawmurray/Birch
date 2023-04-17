/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::trimul;
using numbirch::trimul_grad1;
using numbirch::trimul_grad2;

template<class Left, class Right>
struct TriMul : public Binary<Left,Right> {
  template<class T, class U>
  TriMul(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(trimul)
  BIRCH_BINARY_GRAD(trimul_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
TriMul<Left,Right> trimul(const Left& l, const Right& r) {
  return TriMul<Left,Right>(l, r);
}

}
