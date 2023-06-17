/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<argument Left, argument Middle, argument Right>
struct IBeta {
  BIRCH_TERNARY_FORM(IBeta)
};

BIRCH_TERNARY_SIZE(IBeta)
BIRCH_TERNARY(IBeta, ibeta)

}
