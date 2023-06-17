/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Pow {
  BIRCH_BINARY_FORM(Pow)
};

BIRCH_BINARY_SIZE(Pow)
BIRCH_BINARY(Pow, pow)
BIRCH_BINARY_GRAD(Pow, pow_grad)

}
