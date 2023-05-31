/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Sum {
  BIRCH_UNARY_FORM(Sum, numbirch::sum)
  BIRCH_UNARY_GRAD(numbirch::sum_grad)
  BIRCH_FORM
};

template<class Middle>
auto sum(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::sum(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Sum);
  }
}

}
