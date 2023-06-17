/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct SimulateDirichlet {
  BIRCH_UNARY_FORM(SimulateDirichlet)
};

BIRCH_UNARY_SIZE(SimulateDirichlet)
BIRCH_UNARY(SimulateDirichlet, simulate_dirichlet)

}
