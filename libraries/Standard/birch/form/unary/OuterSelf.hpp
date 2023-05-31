/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct OuterSelf {
  BIRCH_UNARY_FORM(OuterSelf, numbirch::outer)
  BIRCH_UNARY_GRAD(numbirch::outer_grad)
  BIRCH_FORM
};

template<class Middle>
auto outer(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(OuterSelf);
}

}
