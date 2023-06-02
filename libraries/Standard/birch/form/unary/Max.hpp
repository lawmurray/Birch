/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Max {
  BIRCH_UNARY_FORM(Max)
};

BIRCH_UNARY_SIZE(Max)
BIRCH_UNARY(Max, numbirch::max)
BIRCH_UNARY_GRAD(Max, numbirch::max_grad)

template<class Middle>
auto max(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::max(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Max);
  }
}

}
