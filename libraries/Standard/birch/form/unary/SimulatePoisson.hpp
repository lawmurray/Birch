/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct SimulatePoisson {
  BIRCH_UNARY_FORM(SimulatePoisson)
};

BIRCH_UNARY_SIZE(SimulatePoisson)
BIRCH_UNARY(SimulatePoisson, simulate_poisson)

}
