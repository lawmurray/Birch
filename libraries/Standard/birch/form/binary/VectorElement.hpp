/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct VectorElement {
  BIRCH_BINARY_FORM(VectorElement, numbirch::element)
  BIRCH_BINARY_GRAD(numbirch::element_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto element(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(VectorElement);
}

}
