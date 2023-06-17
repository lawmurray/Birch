/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct GammaP {
  BIRCH_BINARY_FORM(GammaP)
};

BIRCH_BINARY_SIZE(GammaP)
BIRCH_BINARY(GammaP, gamma_p)

}
