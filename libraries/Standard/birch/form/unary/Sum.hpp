/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Sum {
  BIRCH_UNARY_FORM(Sum)
  BIRCH_UNARY_SIZE(Sum)
  BIRCH_UNARY_EVAL(Sum, sum)
  BIRCH_UNARY_GRAD(Sum, sum_grad)
};

BIRCH_UNARY_TYPE(Sum)
BIRCH_UNARY_CALL(Sum, sum)

}
