/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct DotSelf {
  BIRCH_UNARY_FORM(DotSelf)
};

BIRCH_UNARY_SIZE(DotSelf)
BIRCH_UNARY(DotSelf, numbirch::dot)
BIRCH_UNARY_GRAD(DotSelf, numbirch::dot_grad)

template<class Middle>
auto dot(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(DotSelf);
}

}
