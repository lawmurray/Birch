/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct TriOuterSelf {
  BIRCH_UNARY_FORM(TriOuterSelf)
};

BIRCH_UNARY_SIZE(TriOuterSelf)
BIRCH_UNARY(TriOuterSelf, numbirch::triouter)
BIRCH_UNARY_GRAD(TriOuterSelf, numbirch::triouter_grad)

template<class Middle>
auto triouter(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(TriOuterSelf);
}

}
