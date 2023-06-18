/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<argument Left, argument Middle, argument Right>
struct Where {
  BIRCH_TERNARY_FORM(Where)
  BIRCH_TERNARY_SIZE(Where)
  BIRCH_TERNARY_EVAL(Where, where)
  BIRCH_TERNARY_GRAD(Where, where_grad)
};

BIRCH_TERNARY_TYPE(Where)
BIRCH_TERNARY_CALL(Where, where)

}
