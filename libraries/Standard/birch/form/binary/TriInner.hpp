/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct TriInner {
  BIRCH_BINARY_FORM(TriInner)
};

BIRCH_BINARY_SIZE(TriInner)
BIRCH_BINARY(TriInner, numbirch::triinner)
BIRCH_BINARY_GRAD(TriInner, numbirch::triinner_grad)

template<class Left, class Right>
auto triinner(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(TriInner);
}

}
