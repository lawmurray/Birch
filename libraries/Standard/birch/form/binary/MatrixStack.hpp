/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct MatrixStack {
  BIRCH_BINARY_FORM(MatrixStack, numbirch::stack)
  BIRCH_BINARY_GRAD(numbirch::stack_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto stack(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(MatrixStack);
}

}
