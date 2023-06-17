/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Neg {
  BIRCH_UNARY_FORM(Neg)
};

BIRCH_UNARY_SIZE(Neg)
BIRCH_UNARY(Neg, neg)
BIRCH_UNARY_GRAD(Neg, neg_grad)

template<argument Middle>
requires (!numbirch::arithmetic<Middle>)
auto operator-(Middle&& m) {
  return neg(std::forward<Middle>(m));
}

}