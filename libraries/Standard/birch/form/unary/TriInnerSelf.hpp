/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct TriInnerSelf {
  BIRCH_UNARY_FORM(TriInnerSelf, numbirch::triinner)
  BIRCH_UNARY_GRAD(numbirch::triinner_grad)
  BIRCH_FORM
};

template<class Middle>
auto triinner(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(TriInnerSelf);
}

}
