/**
 * @file
 */
#pragma once

#include "birch/form/Infix.hpp"

namespace birch {

template<class Left, class Right>
struct Sub {
  BIRCH_BINARY_FORM(Sub, numbirch::sub)
  BIRCH_BINARY_GRAD(numbirch::sub_grad)
  BIRCH_FORM
};

template<class Left, class Right, std::enable_if_t<
    is_numerical_v<Left> && is_numerical_v<Right> &&
    !(numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>),
    int> = 0>
auto operator-(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(Sub);
}

}
