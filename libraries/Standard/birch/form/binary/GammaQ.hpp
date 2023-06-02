/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct GammaQ {
  BIRCH_BINARY_FORM(GammaQ)
};

BIRCH_BINARY_SIZE(GammaQ)
BIRCH_BINARY(GammaQ, numbirch::gamma_q)

template<class Left, class Right>
auto gamma_q(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::gamma_q(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(GammaQ);
  }
}

}
