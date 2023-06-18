/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct LessOrEqual {
  BIRCH_BINARY_FORM(LessOrEqual)
  BIRCH_BINARY_SIZE(LessOrEqual)
  BIRCH_BINARY_EVAL(LessOrEqual, less_or_equal)
  BIRCH_BINARY_GRAD(LessOrEqual, less_or_equal_grad)
};

BIRCH_BINARY_TYPE(LessOrEqual)
BIRCH_BINARY_CALL(LessOrEqual, less_or_equal)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator<=(Left&& l, Right&& r) {
  return less_or_equal(std::forward<Left>(l), std::forward<Right>(r));
}

}
