/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct SimulateDirichlet {
  BIRCH_UNARY_FORM(SimulateDirichlet)
  BIRCH_UNARY_SIZE(SimulateDirichlet)
  BIRCH_UNARY_EVAL(SimulateDirichlet, simulate_dirichlet)
};

BIRCH_UNARY_TYPE(SimulateDirichlet)
BIRCH_UNARY_CALL(SimulateDirichlet, simulate_dirichlet)

}
