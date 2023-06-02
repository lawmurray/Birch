/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct TriMul {
  BIRCH_BINARY_FORM(TriMul)
};

BIRCH_BINARY_SIZE(TriMul)
BIRCH_BINARY(TriMul, numbirch::trimul)
BIRCH_BINARY_GRAD(TriMul, numbirch::trimul_grad)

template<class Left, class Right>
auto trimul(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(TriMul);
}

}
