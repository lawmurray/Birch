/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct LGamma {
  BIRCH_UNARY_FORM(LGamma)
};

BIRCH_UNARY_SIZE(LGamma)
BIRCH_UNARY(LGamma, numbirch::lgamma)
BIRCH_UNARY_GRAD(LGamma, numbirch::lgamma_grad)

template<class Middle>
auto lgamma(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::lgamma(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(LGamma);
  }
}

}
