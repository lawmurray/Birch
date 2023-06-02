/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Cosh {
  BIRCH_UNARY_FORM(Cosh)
};

BIRCH_UNARY_SIZE(Cosh)
BIRCH_UNARY(Cosh, numbirch::cosh)
BIRCH_UNARY_GRAD(Cosh, numbirch::cosh_grad)

template<class Middle>
auto cosh(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::cosh(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Cosh);
  }
}

}
