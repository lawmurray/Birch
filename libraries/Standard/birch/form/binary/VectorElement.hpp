/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct VectorElement {
  BIRCH_BINARY_FORM(VectorElement)
  BIRCH_BINARY_SIZE(VectorElement)
  BIRCH_BINARY_EVAL(VectorElement, element)
  BIRCH_BINARY_GRAD(VectorElement, element_grad)
};

BIRCH_BINARY_TYPE(VectorElement)
BIRCH_BINARY_CALL(VectorElement, element)

}
