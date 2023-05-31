/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Round {
  BIRCH_UNARY_FORM(Round, numbirch::round)
  BIRCH_UNARY_GRAD(numbirch::round_grad)
  BIRCH_FORM
};

template<class Middle>
auto round(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::round(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Round);
  }
}

}
