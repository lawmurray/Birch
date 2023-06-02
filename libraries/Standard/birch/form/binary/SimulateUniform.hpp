/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct SimulateUniform {
  BIRCH_BINARY_FORM(SimulateUniform)
};

BIRCH_BINARY_SIZE(SimulateUniform)
BIRCH_BINARY(SimulateUniform, numbirch::simulate_uniform)

template<class Left, class Right>
auto simulate_uniform(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::simulate_uniform(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(SimulateUniform);
  }
}

}
