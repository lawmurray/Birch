/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct InnerSelf {
  BIRCH_UNARY_FORM(InnerSelf, numbirch::inner)
  BIRCH_UNARY_GRAD(numbirch::inner_grad)
  BIRCH_FORM
};

template<class Middle>
auto inner(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(InnerSelf);
}

}
