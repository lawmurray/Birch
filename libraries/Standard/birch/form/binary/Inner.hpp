/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Inner {
  BIRCH_BINARY_FORM(Inner)
};

BIRCH_BINARY_SIZE(Inner)
BIRCH_BINARY(Inner, inner)
BIRCH_BINARY_GRAD(Inner, inner_grad)

}
