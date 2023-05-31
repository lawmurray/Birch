/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct SimulateWishart {
  BIRCH_UNARY_FORM(SimulateWishart, numbirch::simulate_wishart, n)
  BIRCH_FORM
};

template<class Middle>
auto simulate_wishart(const Middle& m, const int n) {
  return BIRCH_UNARY_CONSTRUCT(SimulateWishart, n);
}

}
