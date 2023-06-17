/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Dot {
  BIRCH_BINARY_FORM(Dot)
};

BIRCH_BINARY_SIZE(Dot)
BIRCH_BINARY(Dot, dot)
BIRCH_BINARY_GRAD(Dot, dot_grad)

}