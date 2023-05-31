/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<class Left, class Middle, class Right>
struct MatrixElement {
  BIRCH_TERNARY_FORM(MatrixElement, numbirch::element)
  BIRCH_TERNARY_GRAD(numbirch::element_grad)
  BIRCH_FORM
};

template<class Left, class Middle, class Right>
auto element(const Left& l, const Middle& m, const Right& r) {
  return BIRCH_TERNARY_CONSTRUCT(MatrixElement);
}

}
