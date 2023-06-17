/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Sinh {
  BIRCH_UNARY_FORM(Sinh)
};

BIRCH_UNARY_SIZE(Sinh)
BIRCH_UNARY(Sinh, sinh)
BIRCH_UNARY_GRAD(Sinh, sinh_grad)

}
