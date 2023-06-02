/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Pos {
  BIRCH_UNARY_FORM(Pos)
};

BIRCH_UNARY_SIZE(Pos)
BIRCH_UNARY(Pos, numbirch::pos)
BIRCH_UNARY_GRAD(Pos, numbirch::pos_grad)

template<class Middle, std::enable_if_t<
    is_numerical_v<Middle> && !numbirch::is_arithmetic_v<Middle>,int> = 0>
auto operator+(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(Pos);
}

}
