/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct Convolve {
  BIRCH_BINARY_FORM(Convolve)
};

BIRCH_BINARY_SIZE(Convolve)
BIRCH_BINARY(Convolve, numbirch::convolve)
BIRCH_BINARY_GRAD(Convolve, numbirch::convolve_grad)

template<class Left, class Right>
auto convolve(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(Convolve);
}

}
