/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Abs {
  BIRCH_UNARY_FORM(Abs)
};

BIRCH_UNARY_SIZE(Abs)
BIRCH_UNARY(Abs, abs)
BIRCH_UNARY_GRAD(Abs, abs_grad)

}
