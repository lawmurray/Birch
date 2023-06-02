/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Not {
  BIRCH_UNARY_FORM(Not)
};

BIRCH_UNARY_SIZE(Not)
BIRCH_UNARY(Not, numbirch::logical_not)
BIRCH_UNARY_GRAD(Not, numbirch::logical_not_grad)

template<class Middle, std::enable_if_t<
    is_numerical_v<Middle> && !numbirch::is_arithmetic_v<Middle>,int> = 0>
auto operator!(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(Not);
}

}
