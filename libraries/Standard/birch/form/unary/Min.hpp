/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Min {
  BIRCH_UNARY_FORM(Min)
};

BIRCH_UNARY_SIZE(Min)
BIRCH_UNARY(Min, min)
BIRCH_UNARY_GRAD(Min, min_grad)

}
