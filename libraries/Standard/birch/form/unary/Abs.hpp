/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Abs {
  BIRCH_UNARY_FORM(Abs, numbirch::abs)
  BIRCH_UNARY_GRAD(numbirch::abs_grad)
  BIRCH_FORM
};

template<class Middle>
auto abs(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::abs(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Abs);
  }
}

}
