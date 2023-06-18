/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct LChoose {
  BIRCH_BINARY_FORM(LChoose)
  BIRCH_BINARY_SIZE(LChoose)
  BIRCH_BINARY_EVAL(LChoose, lchoose)
  BIRCH_BINARY_GRAD(LChoose, lchoose_grad)
};

BIRCH_BINARY_TYPE(LChoose)
BIRCH_BINARY_CALL(LChoose, lchoose)

}
