/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct GreaterOrEqual {
  BIRCH_BINARY_FORM(GreaterOrEqual)
};

BIRCH_BINARY_SIZE(GreaterOrEqual)
BIRCH_BINARY(GreaterOrEqual, greater_or_equal)
BIRCH_BINARY_GRAD(GreaterOrEqual, greater_or_equal_grad)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator>=(Left&& l, Right&& r) {
  return greater_or_equal(std::forward<Left>(l), std::forward<Right>(r));
}

}
