/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct TriOuterSelf {
  BIRCH_UNARY_FORM(TriOuterSelf)
};

BIRCH_UNARY_SIZE(TriOuterSelf)
BIRCH_UNARY(TriOuterSelf, triouter)
BIRCH_UNARY_GRAD(TriOuterSelf, triouter_grad)

}
