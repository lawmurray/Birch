/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<argument Left, argument Middle, argument Right>
struct MatrixGather {
  BIRCH_TERNARY_FORM(MatrixGather)
  BIRCH_TERNARY_SIZE(MatrixGather)
  BIRCH_TERNARY_EVAL(MatrixGather, gather)
  BIRCH_TERNARY_GRAD(MatrixGather, gather_grad)
};

BIRCH_TERNARY_TYPE(MatrixGather)
BIRCH_TERNARY_CALL(MatrixGather, gather)

}
