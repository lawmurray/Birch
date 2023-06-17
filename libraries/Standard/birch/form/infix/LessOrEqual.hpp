/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct LessOrEqual {
  BIRCH_BINARY_FORM(LessOrEqual)
};

BIRCH_BINARY_SIZE(LessOrEqual)
BIRCH_BINARY(LessOrEqual, less_or_equal)
BIRCH_BINARY_GRAD(LessOrEqual, less_or_equal_grad)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator<=(Left&& l, Right&& r) {
  return less_or_equal(std::forward<Left>(l), std::forward<Right>(r));
}

}
