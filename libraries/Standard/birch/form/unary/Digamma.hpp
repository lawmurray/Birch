/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Digamma {
  BIRCH_UNARY_FORM(Digamma)
  BIRCH_UNARY_SIZE(Digamma)
  BIRCH_UNARY_EVAL(Digamma, digamma)
};

BIRCH_UNARY_TYPE(Digamma)
BIRCH_UNARY_CALL(Digamma, digamma)

}
