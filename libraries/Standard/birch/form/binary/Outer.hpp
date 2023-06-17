/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Outer {
  BIRCH_BINARY_FORM(Outer)
};

BIRCH_BINARY_SIZE(Outer)
BIRCH_BINARY(Outer, outer)
BIRCH_BINARY_GRAD(Outer, outer_grad)

}
