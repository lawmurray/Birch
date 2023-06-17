/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Expm1 {
  BIRCH_UNARY_FORM(Expm1)
};

BIRCH_UNARY_SIZE(Expm1)
BIRCH_UNARY(Expm1, expm1)
BIRCH_UNARY_GRAD(Expm1, expm1_grad)

}
