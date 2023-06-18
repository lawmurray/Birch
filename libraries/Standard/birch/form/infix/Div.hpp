/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Div {
  BIRCH_BINARY_FORM(Div)
  BIRCH_BINARY_SIZE(Div)
  BIRCH_BINARY_EVAL(Div, div)
  BIRCH_BINARY_GRAD(Div, div_grad)
};

BIRCH_BINARY_TYPE(Div)
BIRCH_BINARY_CALL(Div, div)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator/(Left&& l, Right&& r) {
  return div(std::forward<Left>(l), std::forward<Right>(r));
}

}
