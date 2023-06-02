/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<class Left, class Middle, class Right>
struct MatrixElement {
  BIRCH_TERNARY_FORM(MatrixElement)
};

BIRCH_TERNARY_SIZE(MatrixElement)
BIRCH_TERNARY(MatrixElement, numbirch::element)
BIRCH_TERNARY_GRAD(MatrixElement, numbirch::element_grad)

template<class Left, class Middle, class Right>
auto element(const Left& l, const Middle& m, const Right& r) {
  return BIRCH_TERNARY_CONSTRUCT(MatrixElement);
}

}
