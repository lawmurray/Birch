/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct Pow {
  BIRCH_BINARY_FORM(Pow)
};

BIRCH_BINARY_SIZE(Pow)
BIRCH_BINARY(Pow, numbirch::pow)
BIRCH_BINARY_GRAD(Pow, numbirch::pow_grad)

template<class Left, class Right>
auto pow(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::pow(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(Pow);
  }
}

}
