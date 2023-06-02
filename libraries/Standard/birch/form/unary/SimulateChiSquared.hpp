/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct SimulateChiSquared {
  BIRCH_UNARY_FORM(SimulateChiSquared)
};

BIRCH_UNARY_SIZE(SimulateChiSquared)
BIRCH_UNARY(SimulateChiSquared, numbirch::simulate_chi_squared)

template<class Middle>
auto simulate_chi_squared(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::simulate_chi_squared(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(SimulateChiSquared);
  }
}

}
