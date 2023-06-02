/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct Dot {
  BIRCH_BINARY_FORM(Dot)
};

BIRCH_BINARY_SIZE(Dot)
BIRCH_BINARY(Dot, numbirch::dot)
BIRCH_BINARY_GRAD(Dot, numbirch::dot_grad)

template<class Left, class Right>
auto dot(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(Dot);
}

}
