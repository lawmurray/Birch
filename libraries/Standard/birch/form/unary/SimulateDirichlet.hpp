/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct SimulateDirichlet {
  BIRCH_UNARY_FORM(SimulateDirichlet)
};

BIRCH_UNARY_SIZE(SimulateDirichlet)
BIRCH_UNARY(SimulateDirichlet, numbirch::simulate_dirichlet)

template<class Middle>
auto simulate_dirichlet(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::simulate_dirichlet(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(SimulateDirichlet);
  }
}

}
