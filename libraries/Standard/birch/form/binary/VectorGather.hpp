/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::gather;
using numbirch::gather_grad1;
using numbirch::gather_grad2;

template<class Left, class Right>
struct VectorGather : public Binary<Left,Right> {
  template<class T, class U>
  VectorGather(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(gather)
  BIRCH_BINARY_GRAD(gather_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
VectorGather<Left,Right> gather(const Left& l, const Right& r) {
  return VectorGather<Left,Right>(l, r);
}

}
