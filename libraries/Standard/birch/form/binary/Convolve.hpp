/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Convolve {
  BIRCH_BINARY_FORM(Convolve)
};

BIRCH_BINARY_SIZE(Convolve)
BIRCH_BINARY(Convolve, convolve)
BIRCH_BINARY_GRAD(Convolve, convolve_grad)

}
