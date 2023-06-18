/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct SimulateWishart {
  BIRCH_UNARY_FORM(SimulateWishart, n)
  BIRCH_UNARY_SIZE(SimulateWishart)
  BIRCH_UNARY_EVAL(SimulateWishart, simulate_wishart, n)
};

BIRCH_UNARY_TYPE(SimulateWishart)
BIRCH_UNARY_CALL(SimulateWishart, simulate_wishart, n)

}
