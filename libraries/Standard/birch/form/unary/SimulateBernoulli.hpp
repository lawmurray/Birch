/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct SimulateBernoulli {
  BIRCH_UNARY_FORM(SimulateBernoulli)
};

BIRCH_UNARY_SIZE(SimulateBernoulli)
BIRCH_UNARY(SimulateBernoulli, simulate_bernoulli)

}