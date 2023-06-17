/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct And {
  BIRCH_BINARY_FORM(And)
};

BIRCH_BINARY_SIZE(And)
BIRCH_BINARY(And, logical_and)
BIRCH_BINARY_GRAD(And, logical_and_grad)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator&&(Left&& l, Right&& r) {
  return logical_and(std::forward<Left>(l), std::forward<Right>(r));
}

}
