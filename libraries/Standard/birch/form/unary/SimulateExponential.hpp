/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct SimulateExponential {
  BIRCH_UNARY_FORM(SimulateExponential)
};

BIRCH_UNARY_SIZE(SimulateExponential)
BIRCH_UNARY(SimulateExponential, numbirch::simulate_exponential)

template<class Middle>
auto simulate_exponential(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::simulate_exponential(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(SimulateExponential);
  }
}

}
