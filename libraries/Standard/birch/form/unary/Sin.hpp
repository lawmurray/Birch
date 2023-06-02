/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Sin {
  BIRCH_UNARY_FORM(Sin)
};

BIRCH_UNARY_SIZE(Sin)
BIRCH_UNARY(Sin, numbirch::sin)
BIRCH_UNARY_GRAD(Sin, numbirch::sin_grad)

template<class Middle>
auto sin(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::sin(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Sin);
  }
}

}
