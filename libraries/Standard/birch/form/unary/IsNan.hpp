/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct IsNan {
  BIRCH_UNARY_FORM(IsNan)
};

BIRCH_UNARY_SIZE(IsNan)
BIRCH_UNARY(IsNan, numbirch::isnan)
BIRCH_UNARY_GRAD(IsNan, numbirch::isnan_grad)

template<class Middle>
auto isnan(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::isnan(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(IsNan);
  }
}

}
