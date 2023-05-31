/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Transpose {
  BIRCH_UNARY_FORM(Transpose, numbirch::transpose)
  BIRCH_UNARY_GRAD(numbirch::transpose_grad)
  BIRCH_FORM
};

template<class Middle>
auto transpose(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::transpose(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Transpose);
  }
}

}
