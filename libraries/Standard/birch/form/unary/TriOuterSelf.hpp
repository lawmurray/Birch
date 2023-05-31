/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct TriOuterSelf {
  BIRCH_UNARY_FORM(TriOuterSelf, numbirch::triouter)
  BIRCH_UNARY_GRAD(numbirch::triouter_grad)
  BIRCH_FORM
};

template<class Middle>
auto triouter(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(TriOuterSelf);
}

}
