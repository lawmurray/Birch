/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Tan {
  BIRCH_UNARY_FORM(Tan, numbirch::tan)
  BIRCH_UNARY_GRAD(numbirch::tan_grad)
  BIRCH_FORM
};

template<class Middle>
auto tan(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::tan(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Tan);
  }
}

}
