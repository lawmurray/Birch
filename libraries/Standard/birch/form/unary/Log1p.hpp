/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Log1p {
  BIRCH_UNARY_FORM(Log1p)
};

BIRCH_UNARY_SIZE(Log1p)
BIRCH_UNARY(Log1p, log1p)
BIRCH_UNARY_GRAD(Log1p, log1p_grad)

}
