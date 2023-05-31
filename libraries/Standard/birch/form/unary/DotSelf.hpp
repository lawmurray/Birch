/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct DotSelf {
  BIRCH_UNARY_FORM(DotSelf, numbirch::dot)
  BIRCH_UNARY_GRAD(numbirch::dot_grad)
  BIRCH_FORM
};

template<class Middle>
auto dot(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(DotSelf);
}

}
