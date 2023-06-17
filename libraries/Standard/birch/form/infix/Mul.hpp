/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Mul {
  BIRCH_BINARY_FORM(Mul)
};

BIRCH_BINARY_SIZE(Mul)
BIRCH_BINARY(Mul, mul)
BIRCH_BINARY_GRAD(Mul, mul_grad)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator*(Left&& l, Right&& r) {
  return mul(std::forward<Left>(l), std::forward<Right>(r));
}

}
