/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Inv {
  BIRCH_UNARY_FORM(Inv)
  BIRCH_UNARY_SIZE(Inv)
  BIRCH_UNARY_EVAL(Inv, inv)
  BIRCH_UNARY_GRAD(Inv, inv_grad)
};

BIRCH_UNARY_TYPE(Inv)
BIRCH_UNARY_CALL(Inv, inv)

}
