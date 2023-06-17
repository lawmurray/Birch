/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Hadamard {
  BIRCH_BINARY_FORM(Hadamard)
};

BIRCH_BINARY_SIZE(Hadamard)
BIRCH_BINARY(Hadamard, hadamard)
BIRCH_BINARY_GRAD(Hadamard, hadamard_grad)

}
