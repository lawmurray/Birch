/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Acos {
  BIRCH_UNARY_FORM(Acos)
};

BIRCH_UNARY_SIZE(Acos)
BIRCH_UNARY(Acos, acos)
BIRCH_UNARY_GRAD(Acos, acos_grad)

}
