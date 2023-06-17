/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct VectorScatter {
  BIRCH_BINARY_FORM(VectorScatter, n)
};

BIRCH_BINARY_SIZE(VectorScatter)
BIRCH_BINARY(VectorScatter, scatter, n)
BIRCH_BINARY_GRAD(VectorScatter, scatter_grad, n)

}
