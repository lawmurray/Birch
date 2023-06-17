/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct TriMul {
  BIRCH_BINARY_FORM(TriMul)
};

BIRCH_BINARY_SIZE(TriMul)
BIRCH_BINARY(TriMul, trimul)
BIRCH_BINARY_GRAD(TriMul, trimul_grad)

}
