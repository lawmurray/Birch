/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Max {
  BIRCH_UNARY_FORM(Max)
};

BIRCH_UNARY_SIZE(Max)
BIRCH_UNARY(Max, max)
BIRCH_UNARY_GRAD(Max, max_grad)

}
