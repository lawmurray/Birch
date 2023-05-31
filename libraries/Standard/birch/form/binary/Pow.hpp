/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct Pow {
  BIRCH_BINARY_FORM(Pow, numbirch::pow)
  BIRCH_BINARY_GRAD(numbirch::pow_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto pow(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::pow(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(Pow);
  }
}

}
