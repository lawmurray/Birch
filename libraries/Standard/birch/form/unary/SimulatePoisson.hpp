/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct SimulatePoisson {
  BIRCH_UNARY_FORM(SimulatePoisson, numbirch::simulate_poisson)
  BIRCH_FORM
};

template<class Middle>
auto simulate_poisson(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::simulate_poisson(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(SimulatePoisson);
  }
}

}
