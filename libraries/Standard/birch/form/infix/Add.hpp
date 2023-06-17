/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Add {
  BIRCH_BINARY_FORM(Add)
};

BIRCH_BINARY_SIZE(Add)
BIRCH_BINARY(Add, add)
BIRCH_BINARY_GRAD(Add, add_grad)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator+(Left&& l, Right&& r) {
  return add(std::forward<Left>(l), std::forward<Right>(r));
}

}
