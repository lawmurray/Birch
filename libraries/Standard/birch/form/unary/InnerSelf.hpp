/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct InnerSelf {
  BIRCH_UNARY_FORM(InnerSelf)
};

BIRCH_UNARY_SIZE(InnerSelf)
BIRCH_UNARY(InnerSelf, numbirch::inner)
BIRCH_UNARY_GRAD(InnerSelf, numbirch::inner_grad)

template<class Middle>
auto inner(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(InnerSelf);
}

}
