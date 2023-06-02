/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct VectorElement {
  BIRCH_BINARY_FORM(VectorElement)
};

BIRCH_BINARY_SIZE(VectorElement)
BIRCH_BINARY(VectorElement, numbirch::element)
BIRCH_BINARY_GRAD(VectorElement, numbirch::element_grad)

template<class Left, class Right>
auto element(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(VectorElement);
}

}
