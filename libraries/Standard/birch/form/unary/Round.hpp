/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Round {
  BIRCH_UNARY_FORM(Round)
};

BIRCH_UNARY_SIZE(Round)
BIRCH_UNARY(Round, round)
BIRCH_UNARY_GRAD(Round, round_grad)

}
