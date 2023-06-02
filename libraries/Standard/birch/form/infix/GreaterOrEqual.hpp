/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct GreaterOrEqual {
  BIRCH_BINARY_FORM(GreaterOrEqual)
};

BIRCH_BINARY_SIZE(GreaterOrEqual)
BIRCH_BINARY(GreaterOrEqual, numbirch::greater_or_equal)
BIRCH_BINARY_GRAD(GreaterOrEqual, numbirch::greater_or_equal_grad)

template<class Left, class Right, std::enable_if_t<
    is_numerical_v<Left> && is_numerical_v<Right> &&
    !(numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>),
    int> = 0>
auto operator>=(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(GreaterOrEqual);
}

}
