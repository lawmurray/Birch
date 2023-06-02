/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct LBeta {
  BIRCH_BINARY_FORM(LBeta)
};

BIRCH_BINARY_SIZE(LBeta)
BIRCH_BINARY(LBeta, numbirch::lbeta)
BIRCH_BINARY_GRAD(LBeta, numbirch::lbeta_grad)

template<class Left, class Right>
auto lbeta(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::lbeta(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(LBeta);
  }
}

}
