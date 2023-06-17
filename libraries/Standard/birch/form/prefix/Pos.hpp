/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Pos {
  BIRCH_UNARY_FORM(Pos)
};

BIRCH_UNARY_SIZE(Pos)
BIRCH_UNARY(Pos, pos)
BIRCH_UNARY_GRAD(Pos, pos_grad)

template<argument Middle>
requires (!numbirch::arithmetic<Middle>)
auto operator+(Middle&& m) {
  return pos(std::forward<Middle>(m));
}

}