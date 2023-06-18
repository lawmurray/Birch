/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Asin {
  BIRCH_UNARY_FORM(Asin)
  BIRCH_UNARY_SIZE(Asin)
  BIRCH_UNARY_EVAL(Asin, asin)
  BIRCH_UNARY_GRAD(Asin, asin_grad)
};

BIRCH_UNARY_TYPE(Asin)
BIRCH_UNARY_CALL(Asin, asin)

}
