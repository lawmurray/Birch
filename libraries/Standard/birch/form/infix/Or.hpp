/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Or {
  BIRCH_BINARY_FORM(Or)
  BIRCH_BINARY_SIZE(Or)
  BIRCH_BINARY_EVAL(Or, logical_or)
  BIRCH_BINARY_GRAD(Or, logical_or_grad)
};

BIRCH_BINARY_TYPE(Or)
BIRCH_BINARY_CALL(Or, logical_or)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator||(Left&& l, Right&& r) {
  return logical_or(std::forward<Left>(l), std::forward<Right>(r));
}

}
