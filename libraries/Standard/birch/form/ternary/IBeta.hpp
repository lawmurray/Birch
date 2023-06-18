/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<argument Left, argument Middle, argument Right>
struct IBeta {
  BIRCH_TERNARY_FORM(IBeta)
  BIRCH_TERNARY_SIZE(IBeta)
  BIRCH_TERNARY_EVAL(IBeta, ibeta)
};

BIRCH_TERNARY_TYPE(IBeta)
BIRCH_TERNARY_CALL(IBeta, ibeta)

}
