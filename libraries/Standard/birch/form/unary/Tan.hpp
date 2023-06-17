/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Tan {
  BIRCH_UNARY_FORM(Tan)
};

BIRCH_UNARY_SIZE(Tan)
BIRCH_UNARY(Tan, tan)
BIRCH_UNARY_GRAD(Tan, tan_grad)

}
