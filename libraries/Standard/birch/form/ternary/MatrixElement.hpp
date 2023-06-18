/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<argument Left, argument Middle, argument Right>
struct MatrixElement {
  BIRCH_TERNARY_FORM(MatrixElement)
  BIRCH_TERNARY_SIZE(MatrixElement)
  BIRCH_TERNARY_EVAL(MatrixElement, element)
  BIRCH_TERNARY_GRAD(MatrixElement, element_grad)
};

BIRCH_TERNARY_TYPE(MatrixElement)
BIRCH_TERNARY_CALL(MatrixElement, element)

}
