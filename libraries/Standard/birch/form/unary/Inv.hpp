/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Inv {
  BIRCH_UNARY_FORM(Inv, numbirch::inv)
  BIRCH_UNARY_GRAD(numbirch::inv_grad)
  BIRCH_FORM
};

template<class Middle>
auto inv(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::inv(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Inv);
  }
}

}
