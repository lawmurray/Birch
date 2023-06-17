/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct SimulateExponential {
  BIRCH_UNARY_FORM(SimulateExponential)
};

BIRCH_UNARY_SIZE(SimulateExponential)
BIRCH_UNARY(SimulateExponential, simulate_exponential)

}
