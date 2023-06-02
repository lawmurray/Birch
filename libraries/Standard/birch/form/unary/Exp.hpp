/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Exp {
  BIRCH_UNARY_FORM(Exp)
};

BIRCH_UNARY_SIZE(Exp)
BIRCH_UNARY(Exp, numbirch::exp)
BIRCH_UNARY_GRAD(Exp, numbirch::exp_grad)

template<class Middle>
auto exp(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::exp(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Exp);
  }
}

}
