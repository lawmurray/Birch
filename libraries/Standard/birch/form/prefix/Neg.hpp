/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Neg {
  BIRCH_UNARY_FORM(Neg)
};

BIRCH_UNARY_SIZE(Neg)
BIRCH_UNARY(Neg, numbirch::neg)
BIRCH_UNARY_GRAD(Neg, numbirch::neg_grad)

template<class Middle, std::enable_if_t<
    is_numerical_v<Middle> && !numbirch::is_arithmetic_v<Middle>,int> = 0>
auto operator-(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(Neg);
}

}
