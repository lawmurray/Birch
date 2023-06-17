/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Scal {
  BIRCH_UNARY_FORM(Scal)
};

BIRCH_UNARY_SIZE(Scal)
BIRCH_UNARY(Scal, scal)
BIRCH_UNARY_GRAD(Scal, scal_grad)

}
