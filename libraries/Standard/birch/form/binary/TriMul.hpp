/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct TriMul {
  BIRCH_BINARY_FORM(TriMul, numbirch::trimul)
  BIRCH_BINARY_GRAD(numbirch::trimul_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto trimul(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(TriMul);
}

}
