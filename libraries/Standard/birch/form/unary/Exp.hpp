/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Exp {
  BIRCH_UNARY_FORM(Exp)
};

BIRCH_UNARY_SIZE(Exp)
BIRCH_UNARY(Exp, exp)
BIRCH_UNARY_GRAD(Exp, exp_grad)

}
