/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::pack;
using numbirch::pack_grad1;
using numbirch::pack_grad2;

template<class Left, class Right>
struct MatrixPack : public Binary<Left,Right> {
  template<class T, class U>
  MatrixPack(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(pack)
  BIRCH_BINARY_GRAD(pack_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
MatrixPack<Left,Right> pack(const Left& l, const Right& r) {
  return MatrixPack<Left,Right>(l, r);
}

}
