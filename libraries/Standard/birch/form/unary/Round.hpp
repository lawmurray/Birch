/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Round {
  BIRCH_UNARY_FORM(Round)
  BIRCH_UNARY_SIZE(Round)
  BIRCH_UNARY_EVAL(Round, round)
  BIRCH_UNARY_GRAD(Round, round_grad)
};

BIRCH_UNARY_TYPE(Round)
BIRCH_UNARY_CALL(Round, round)

}
