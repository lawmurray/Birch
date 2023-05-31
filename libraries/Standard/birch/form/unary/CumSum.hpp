/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct CumSum {
  BIRCH_UNARY_FORM(CumSum, numbirch::cumsum)
  BIRCH_UNARY_GRAD(numbirch::cumsum_grad)
  BIRCH_FORM
};

template<class Middle>
auto cumsum(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::cumsum(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(CumSum);
  }
}

}
