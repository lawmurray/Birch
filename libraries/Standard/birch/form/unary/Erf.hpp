/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Erf {
  BIRCH_UNARY_FORM(Erf)
};

BIRCH_UNARY_SIZE(Erf)
BIRCH_UNARY(Erf, erf)
BIRCH_UNARY_GRAD(Erf, erf_grad)

}
