/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct GammaQ {
  BIRCH_BINARY_FORM(GammaQ)
};

BIRCH_BINARY_SIZE(GammaQ)
BIRCH_BINARY(GammaQ, gamma_q)

}
