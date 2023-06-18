/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct SimulatePoisson {
  BIRCH_UNARY_FORM(SimulatePoisson)
  BIRCH_UNARY_SIZE(SimulatePoisson)
  BIRCH_UNARY_EVAL(SimulatePoisson, simulate_poisson)
};

BIRCH_UNARY_TYPE(SimulatePoisson)
BIRCH_UNARY_CALL(SimulatePoisson, simulate_poisson)

}
