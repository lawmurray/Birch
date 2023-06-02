/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct OuterSelf {
  BIRCH_UNARY_FORM(OuterSelf)
};

BIRCH_UNARY_SIZE(OuterSelf)
BIRCH_UNARY(OuterSelf, numbirch::outer)
BIRCH_UNARY_GRAD(OuterSelf, numbirch::outer_grad)

template<class Middle>
auto outer(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(OuterSelf);
}

}
