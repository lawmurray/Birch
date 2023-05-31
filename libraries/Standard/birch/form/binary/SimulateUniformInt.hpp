/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct SimulateUniformInt {
  BIRCH_BINARY_FORM(SimulateUniformInt, numbirch::simulate_uniform_int)
  BIRCH_FORM
};

template<class Left, class Right>
auto simulate_uniform_int(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::simulate_uniform_int(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(SimulateUniformInt);
  }
}

}
