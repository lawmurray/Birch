/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Sqrt {
  BIRCH_UNARY_FORM(Sqrt, numbirch::sqrt)
  BIRCH_UNARY_GRAD(numbirch::sqrt_grad)
  BIRCH_FORM
};

template<class Middle>
auto sqrt(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::sqrt(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Sqrt);
  }
}

}
