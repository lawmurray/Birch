/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct LGammaP {
  BIRCH_BINARY_FORM(LGammaP)
};

BIRCH_BINARY_SIZE(LGammaP)
BIRCH_BINARY(LGammaP, numbirch::lgamma)
BIRCH_BINARY_GRAD(LGammaP, numbirch::lgamma_grad)

template<class Left, class Right>
auto lgamma(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::lgamma(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(LGammaP);
  }
}

}
