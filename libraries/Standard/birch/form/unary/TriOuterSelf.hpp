/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct TriOuterSelf {
  BIRCH_UNARY_FORM(TriOuterSelf)
  BIRCH_UNARY_SIZE(TriOuterSelf)
  BIRCH_UNARY_EVAL(TriOuterSelf, triouter)
  BIRCH_UNARY_GRAD(TriOuterSelf, triouter_grad)
};

BIRCH_UNARY_TYPE(TriOuterSelf)
BIRCH_UNARY_CALL(TriOuterSelf, triouter)

}
