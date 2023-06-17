/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct MatrixPack {
  BIRCH_BINARY_FORM(MatrixPack)
};

BIRCH_BINARY_SIZE(MatrixPack)
BIRCH_BINARY(MatrixPack, pack)
BIRCH_BINARY_GRAD(MatrixPack, pack_grad)

}
