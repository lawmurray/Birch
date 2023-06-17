/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct SimulateWishart {
  BIRCH_UNARY_FORM(SimulateWishart, n)
};

BIRCH_UNARY_SIZE(SimulateWishart)
BIRCH_UNARY(SimulateWishart, simulate_wishart, n)

}
