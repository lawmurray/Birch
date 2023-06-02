/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Ceil {
  BIRCH_UNARY_FORM(Ceil)
};

BIRCH_UNARY_SIZE(Ceil)
BIRCH_UNARY(Ceil, numbirch::ceil)
BIRCH_UNARY_GRAD(Ceil, numbirch::ceil_grad)

template<class Middle>
auto ceil(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::ceil(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Ceil);
  }
}

}
