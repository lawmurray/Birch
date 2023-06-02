/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Tan {
  BIRCH_UNARY_FORM(Tan)
};

BIRCH_UNARY_SIZE(Tan)
BIRCH_UNARY(Tan, numbirch::tan)
BIRCH_UNARY_GRAD(Tan, numbirch::tan_grad)

template<class Middle>
auto tan(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::tan(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Tan);
  }
}

}
