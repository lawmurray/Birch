/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct Inner {
  BIRCH_BINARY_FORM(Inner)
};

BIRCH_BINARY_SIZE(Inner)
BIRCH_BINARY(Inner, numbirch::inner)
BIRCH_BINARY_GRAD(Inner, numbirch::inner_grad)

template<class Left, class Right>
auto inner(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(Inner);
}

}
