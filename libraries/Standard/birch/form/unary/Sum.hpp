/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Sum {
  BIRCH_UNARY_FORM(Sum)
};

BIRCH_UNARY_SIZE(Sum)
BIRCH_UNARY(Sum, numbirch::sum)
BIRCH_UNARY_GRAD(Sum, numbirch::sum_grad)

template<class Middle>
auto sum(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::sum(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Sum);
  }
}

}
