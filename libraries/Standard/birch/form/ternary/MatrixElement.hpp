/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<argument Left, argument Middle, argument Right>
struct MatrixElement {
  BIRCH_TERNARY_FORM(MatrixElement)
};

BIRCH_TERNARY_SIZE(MatrixElement)
BIRCH_TERNARY(MatrixElement, element)
BIRCH_TERNARY_GRAD(MatrixElement, element_grad)

}
