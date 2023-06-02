/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct Hadamard {
  BIRCH_BINARY_FORM(Hadamard)
};

BIRCH_BINARY_SIZE(Hadamard)
BIRCH_BINARY(Hadamard, numbirch::hadamard)
BIRCH_BINARY_GRAD(Hadamard, numbirch::hadamard_grad)

template<class Left, class Right>
auto hadamard(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::hadamard(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(Hadamard);
  }
}

}
