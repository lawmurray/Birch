/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct SimulatePoisson {
  BIRCH_UNARY_FORM(SimulatePoisson)
};

BIRCH_UNARY_SIZE(SimulatePoisson)
BIRCH_UNARY(SimulatePoisson, numbirch::simulate_poisson)

template<class Middle>
auto simulate_poisson(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::simulate_poisson(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(SimulatePoisson);
  }
}

}
