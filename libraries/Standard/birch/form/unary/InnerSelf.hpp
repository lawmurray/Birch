/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct InnerSelf {
  BIRCH_UNARY_FORM(InnerSelf)
};

BIRCH_UNARY_SIZE(InnerSelf)
BIRCH_UNARY(InnerSelf, inner)
BIRCH_UNARY_GRAD(InnerSelf, inner_grad)

}
