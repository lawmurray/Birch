/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Sub {
  BIRCH_BINARY_FORM(Sub)
};

BIRCH_BINARY_SIZE(Sub)
BIRCH_BINARY(Sub, sub)
BIRCH_BINARY_GRAD(Sub, sub_grad)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator-(Left&& l, Right&& r) {
  return sub(std::forward<Left>(l), std::forward<Right>(r));
}

}
