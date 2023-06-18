/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Count {
  BIRCH_UNARY_FORM(Count)
  BIRCH_UNARY_SIZE(Count)
  BIRCH_UNARY_EVAL(Count, count)
  BIRCH_UNARY_GRAD(Count, count_grad)
};

BIRCH_UNARY_TYPE(Count)
BIRCH_UNARY_CALL(Count, count)

}
