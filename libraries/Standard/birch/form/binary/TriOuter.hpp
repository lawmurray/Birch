/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct TriOuter {
  BIRCH_BINARY_FORM(TriOuter)
  BIRCH_BINARY_SIZE(TriOuter)
  BIRCH_BINARY_EVAL(TriOuter, triouter)
  BIRCH_BINARY_GRAD(TriOuter, triouter_grad)
};

BIRCH_BINARY_TYPE(TriOuter)
BIRCH_BINARY_CALL(TriOuter, triouter)

}
