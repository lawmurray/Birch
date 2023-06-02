/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct GammaP {
  BIRCH_BINARY_FORM(GammaP)
};

BIRCH_BINARY_SIZE(GammaP)
BIRCH_BINARY(GammaP, numbirch::gamma_p)

template<class Left, class Right>
auto gamma_p(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::gamma_p(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(GammaP);
  }
}

}
