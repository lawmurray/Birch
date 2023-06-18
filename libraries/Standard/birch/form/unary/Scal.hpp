/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Scal {
  BIRCH_UNARY_FORM(Scal)
  BIRCH_UNARY_SIZE(Scal)
  BIRCH_UNARY_EVAL(Scal, scal)
  BIRCH_UNARY_GRAD(Scal, scal_grad)
};

BIRCH_UNARY_TYPE(Scal)
BIRCH_UNARY_CALL(Scal, scal)

}
