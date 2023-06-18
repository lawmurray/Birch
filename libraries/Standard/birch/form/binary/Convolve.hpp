/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Convolve {
  BIRCH_BINARY_FORM(Convolve)
  BIRCH_BINARY_SIZE(Convolve)
  BIRCH_BINARY_EVAL(Convolve, convolve)
  BIRCH_BINARY_GRAD(Convolve, convolve_grad)
};

BIRCH_BINARY_TYPE(Convolve)
BIRCH_BINARY_CALL(Convolve, convolve)

}
