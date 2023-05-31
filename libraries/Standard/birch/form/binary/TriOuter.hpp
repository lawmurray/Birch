/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct TriOuter {
  BIRCH_BINARY_FORM(TriOuter, numbirch::triouter)
  BIRCH_BINARY_GRAD(numbirch::triouter_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto triouter(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(TriOuter);
}

}
