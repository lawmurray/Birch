/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct SimulateGaussian {
  BIRCH_BINARY_FORM(SimulateGaussian, numbirch::simulate_gaussian)
  BIRCH_FORM
};

template<class Left, class Right>
auto simulate_gaussian(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::simulate_gaussian(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(SimulateGaussian);
  }
}

}
