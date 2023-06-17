/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct SimulateChiSquared {
  BIRCH_UNARY_FORM(SimulateChiSquared)
};

BIRCH_UNARY_SIZE(SimulateChiSquared)
BIRCH_UNARY(SimulateChiSquared, simulate_chi_squared)

}
