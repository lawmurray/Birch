/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct TriOuter {
  BIRCH_BINARY_FORM(TriOuter)
};

BIRCH_BINARY_SIZE(TriOuter)
BIRCH_BINARY(TriOuter, numbirch::triouter)
BIRCH_BINARY_GRAD(TriOuter, numbirch::triouter_grad)

template<class Left, class Right>
auto triouter(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(TriOuter);
}

}
