/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Frobenius {
  BIRCH_BINARY_FORM(Frobenius)
};

BIRCH_BINARY_SIZE(Frobenius)
BIRCH_BINARY(Frobenius, frobenius)
BIRCH_BINARY_GRAD(Frobenius, frobenius_grad)

}
