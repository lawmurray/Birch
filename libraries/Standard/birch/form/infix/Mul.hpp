/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct Mul {
  BIRCH_BINARY_FORM(Mul)
};

BIRCH_BINARY_SIZE(Mul)
BIRCH_BINARY(Mul, numbirch::mul)
BIRCH_BINARY_GRAD(Mul, numbirch::mul_grad)

template<class Left, class Right, std::enable_if_t<
    is_numerical_v<Left> && is_numerical_v<Right> &&
    !(numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>),
    int> = 0>
auto operator*(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(Mul);
}

}
