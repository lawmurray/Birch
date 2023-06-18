/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct DigammaP {
  BIRCH_BINARY_FORM(DigammaP)
  BIRCH_BINARY_SIZE(DigammaP)
  BIRCH_BINARY_EVAL(DigammaP, digamma)
};

BIRCH_BINARY_TYPE(DigammaP)
BIRCH_BINARY_CALL(DigammaP, digamma)

}
