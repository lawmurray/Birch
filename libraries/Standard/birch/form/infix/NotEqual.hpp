/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct NotEqual {
  BIRCH_BINARY_FORM(NotEqual)
  BIRCH_BINARY_SIZE(NotEqual)
  BIRCH_BINARY_EVAL(NotEqual, not_equal)
  BIRCH_BINARY_GRAD(NotEqual, not_equal_grad)
};

BIRCH_BINARY_TYPE(NotEqual)
BIRCH_BINARY_CALL(NotEqual, not_equal)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator!=(Left&& l, Right&& r) {
  return not_equal(std::forward<Left>(l), std::forward<Right>(r));
}

}
