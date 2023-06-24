/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"
#include "birch/form/Nullary.hpp"

namespace birch {

template<argument Left, argument Right>
struct DigammaP {
  BIRCH_BINARY_FORM(DigammaP)
  BIRCH_BINARY_SIZE(DigammaP)
  BIRCH_BINARY_EVAL(DigammaP, digamma)
  BIRCH_NULLARY_GRAD(DigammaP)
};

BIRCH_BINARY_TYPE(DigammaP)
BIRCH_BINARY_CALL(DigammaP, digamma)

}
