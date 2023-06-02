/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct TriInnerSelf {
  BIRCH_UNARY_FORM(TriInnerSelf)
};

BIRCH_UNARY_SIZE(TriInnerSelf)
BIRCH_UNARY(TriInnerSelf, numbirch::triinner)
BIRCH_UNARY_GRAD(TriInnerSelf, numbirch::triinner_grad)

template<class Middle>
auto triinner(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(TriInnerSelf);
}

}
