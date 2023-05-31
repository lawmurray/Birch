/**
 * @file
 */
#pragma once

#include "birch/form/Infix.hpp"

namespace birch {

template<class Left, class Right>
struct NotEqual {
  BIRCH_BINARY_FORM(NotEqual, numbirch::not_equal)
  BIRCH_BINARY_GRAD(numbirch::not_equal_grad)
  BIRCH_FORM
};

template<class Left, class Right, std::enable_if_t<
    is_numerical_v<Left> && is_numerical_v<Right> &&
    !(numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>),
    int> = 0>
auto operator!=(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(NotEqual);
}

}
