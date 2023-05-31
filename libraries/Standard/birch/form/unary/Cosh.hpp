/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Cosh {
  BIRCH_UNARY_FORM(Cosh, numbirch::cosh)
  BIRCH_UNARY_GRAD(numbirch::cosh_grad)
  BIRCH_FORM
};

template<class Middle>
auto cosh(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::cosh(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Cosh);
  }
}

}
