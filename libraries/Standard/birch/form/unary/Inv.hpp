/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Inv {
  BIRCH_UNARY_FORM(Inv)
};

BIRCH_UNARY_SIZE(Inv)
BIRCH_UNARY(Inv, numbirch::inv)
BIRCH_UNARY_GRAD(Inv, numbirch::inv_grad)

template<class Middle>
auto inv(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::inv(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Inv);
  }
}

}
