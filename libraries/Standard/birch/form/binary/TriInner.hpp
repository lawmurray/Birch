/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct TriInner {
  BIRCH_BINARY_FORM(TriInner, numbirch::triinner)
  BIRCH_BINARY_GRAD(numbirch::triinner_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto triinner(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(TriInner);
}

}
