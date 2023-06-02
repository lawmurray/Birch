/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct MatrixPack {
  BIRCH_BINARY_FORM(MatrixPack)
};

BIRCH_BINARY_SIZE(MatrixPack)
BIRCH_BINARY(MatrixPack, numbirch::pack)
BIRCH_BINARY_GRAD(MatrixPack, numbirch::pack_grad)

template<class Left, class Right>
auto pack(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(MatrixPack);
}

}
