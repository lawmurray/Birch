/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct DigammaP {
  BIRCH_BINARY_FORM(DigammaP, numbirch::digamma)
  BIRCH_NO_GRAD
  BIRCH_FORM
};

template<class Left, class Right>
auto digamma(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::digamma(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(DigammaP);
  }
}

}
