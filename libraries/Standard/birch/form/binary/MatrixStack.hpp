/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct MatrixStack {
  BIRCH_BINARY_FORM(MatrixStack)
};

BIRCH_BINARY_SIZE(MatrixStack)
BIRCH_BINARY(MatrixStack, numbirch::stack)
BIRCH_BINARY_GRAD(MatrixStack, numbirch::stack_grad)

template<class Left, class Right>
auto stack(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(MatrixStack);
}

}
