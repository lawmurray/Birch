/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct CopySign {
  BIRCH_BINARY_FORM(CopySign)
};

BIRCH_BINARY_SIZE(CopySign)
BIRCH_BINARY(CopySign, numbirch::copysign)
BIRCH_BINARY_GRAD(CopySign, numbirch::copysign_grad)

template<class Left, class Right>
auto copysign(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::copysign(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(CopySign);
  }
}

}
