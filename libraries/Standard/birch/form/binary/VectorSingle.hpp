/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct VectorSingle {
  BIRCH_BINARY_FORM(VectorSingle, numbirch::single, n)
  BIRCH_BINARY_GRAD(numbirch::single_grad, n)
  BIRCH_FORM
};

template<class Left, class Right>
auto single(const Left& l, const Right& r, const int n) {
  return BIRCH_BINARY_CONSTRUCT(VectorSingle, n);
}

}
