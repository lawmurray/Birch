/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Min {
  BIRCH_UNARY_FORM(Min)
};

BIRCH_UNARY_SIZE(Min)
BIRCH_UNARY(Min, numbirch::min)
BIRCH_UNARY_GRAD(Min, numbirch::min_grad)

template<class Middle>
auto min(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::min(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Min);
  }
}

}
