/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Frobenius {
  BIRCH_BINARY_FORM(Frobenius)
  BIRCH_BINARY_SIZE(Frobenius)
  BIRCH_BINARY_EVAL(Frobenius, frobenius)
  BIRCH_BINARY_GRAD(Frobenius, frobenius_grad)
};

BIRCH_BINARY_TYPE(Frobenius)
BIRCH_BINARY_CALL(Frobenius, frobenius)

}
