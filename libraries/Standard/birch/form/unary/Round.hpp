/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Round {
  BIRCH_UNARY_FORM(Round)
};

BIRCH_UNARY_SIZE(Round)
BIRCH_UNARY(Round, numbirch::round)
BIRCH_UNARY_GRAD(Round, numbirch::round_grad)

template<class Middle>
auto round(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::round(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Round);
  }
}

}
