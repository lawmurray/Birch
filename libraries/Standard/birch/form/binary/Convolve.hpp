/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct Convolve {
  BIRCH_BINARY_FORM(Convolve, numbirch::convolve)
  BIRCH_BINARY_GRAD(numbirch::convolve_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto convolve(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(Convolve);
}

}
