/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct CopySign {
  BIRCH_BINARY_FORM(CopySign)
};

BIRCH_BINARY_SIZE(CopySign)
BIRCH_BINARY(CopySign, copysign)
BIRCH_BINARY_GRAD(CopySign, copysign_grad)

}
