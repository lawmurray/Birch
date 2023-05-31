/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct MatrixPack {
  BIRCH_BINARY_FORM(MatrixPack, numbirch::pack)
  BIRCH_BINARY_GRAD(numbirch::pack_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto pack(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(MatrixPack);
}

}
