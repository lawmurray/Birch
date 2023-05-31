/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct VectorGather {
  BIRCH_BINARY_FORM(VectorGather, numbirch::gather)
  BIRCH_BINARY_GRAD(numbirch::gather_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto gather(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(VectorGather);
}

}
