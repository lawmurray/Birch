/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct TriInner {
  BIRCH_BINARY_FORM(TriInner)
};

BIRCH_BINARY_SIZE(TriInner)
BIRCH_BINARY(TriInner, triinner)
BIRCH_BINARY_GRAD(TriInner, triinner_grad)

}
