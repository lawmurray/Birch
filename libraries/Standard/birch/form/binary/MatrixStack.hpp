/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct MatrixStack {
  BIRCH_BINARY_FORM(MatrixStack)
  BIRCH_BINARY_SIZE(MatrixStack)
  BIRCH_BINARY_EVAL(MatrixStack, stack)
  BIRCH_BINARY_GRAD(MatrixStack, stack_grad)
};

BIRCH_BINARY_TYPE(MatrixStack)
BIRCH_BINARY_CALL(MatrixStack, stack)

}
