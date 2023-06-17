/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Greater {
  BIRCH_BINARY_FORM(Greater)
};

BIRCH_BINARY_SIZE(Greater)
BIRCH_BINARY(Greater, greater)
BIRCH_BINARY_GRAD(Greater, greater_grad)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator>(Left&& l, Right&& r) {
  return greater(std::forward<Left>(l), std::forward<Right>(r));
}

}
