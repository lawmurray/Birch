/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Less {
  BIRCH_BINARY_FORM(Less)
  BIRCH_BINARY_SIZE(Less)
  BIRCH_BINARY_EVAL(Less, less)
  BIRCH_BINARY_GRAD(Less, less_grad)
};

BIRCH_BINARY_TYPE(Less)
BIRCH_BINARY_CALL(Less, less)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator<(Left&& l, Right&& r) {
  return less(std::forward<Left>(l), std::forward<Right>(r));
}

}
