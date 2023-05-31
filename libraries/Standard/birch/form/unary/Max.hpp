/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Max {
  BIRCH_UNARY_FORM(Max, numbirch::max)
  BIRCH_UNARY_GRAD(numbirch::max_grad)
  BIRCH_FORM
};

template<class Middle>
auto max(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::max(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Max);
  }
}

}
