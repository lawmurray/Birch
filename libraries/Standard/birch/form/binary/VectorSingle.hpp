/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct VectorSingle {
  BIRCH_BINARY_FORM(VectorSingle, n)
};

BIRCH_BINARY_SIZE(VectorSingle)
BIRCH_BINARY(VectorSingle, single, n)
BIRCH_BINARY_GRAD(VectorSingle, single_grad, n)

}
