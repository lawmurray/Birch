/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Rectify {
  BIRCH_UNARY_FORM(Rectify)
};

BIRCH_UNARY_SIZE(Rectify)
BIRCH_UNARY(Rectify, numbirch::rectify)
BIRCH_UNARY_GRAD(Rectify, numbirch::rectify_grad)

template<class Middle>
auto rectify(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::rectify(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Rectify);
  }
}

}
