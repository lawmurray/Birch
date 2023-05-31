/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct Outer {
  BIRCH_BINARY_FORM(Outer, numbirch::outer)
  BIRCH_BINARY_GRAD(numbirch::outer_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto outer(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(Outer);
}

}
