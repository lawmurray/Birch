/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Ceil {
  BIRCH_UNARY_FORM(Ceil, numbirch::ceil)
  BIRCH_UNARY_GRAD(numbirch::ceil_grad)
  BIRCH_FORM
};

template<class Middle>
auto ceil(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::ceil(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Ceil);
  }
}

}
