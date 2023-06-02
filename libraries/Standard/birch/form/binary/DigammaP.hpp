/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct DigammaP {
  BIRCH_BINARY_FORM(DigammaP)
};

BIRCH_BINARY_SIZE(DigammaP)
BIRCH_BINARY(DigammaP, numbirch::digamma)

template<class Left, class Right>
auto digamma(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::digamma(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(DigammaP);
  }
}

}
