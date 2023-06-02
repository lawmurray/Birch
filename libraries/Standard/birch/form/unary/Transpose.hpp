/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Transpose {
  BIRCH_UNARY_FORM(Transpose)
};

BIRCH_UNARY_SIZE(Transpose)
BIRCH_UNARY(Transpose, numbirch::transpose)
BIRCH_UNARY_GRAD(Transpose, numbirch::transpose_grad)

template<class Middle>
auto transpose(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::transpose(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Transpose);
  }
}

}
