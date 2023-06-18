/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct VectorGather {
  BIRCH_BINARY_FORM(VectorGather)
  BIRCH_BINARY_SIZE(VectorGather)
  BIRCH_BINARY_EVAL(VectorGather, gather)
  BIRCH_BINARY_GRAD(VectorGather, gather_grad)
};

BIRCH_BINARY_TYPE(VectorGather)
BIRCH_BINARY_CALL(VectorGather, gather)

}
