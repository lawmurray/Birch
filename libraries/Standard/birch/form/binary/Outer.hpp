/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct Outer {
  BIRCH_BINARY_FORM(Outer)
};

BIRCH_BINARY_SIZE(Outer)
BIRCH_BINARY(Outer, numbirch::outer)
BIRCH_BINARY_GRAD(Outer, numbirch::outer_grad)

template<class Left, class Right>
auto outer(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(Outer);
}

}
