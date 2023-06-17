/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct TriInnerSelf {
  BIRCH_UNARY_FORM(TriInnerSelf)
};

BIRCH_UNARY_SIZE(TriInnerSelf)
BIRCH_UNARY(TriInnerSelf, triinner)
BIRCH_UNARY_GRAD(TriInnerSelf, triinner_grad)

}
