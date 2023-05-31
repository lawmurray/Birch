/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Rectify {
  BIRCH_UNARY_FORM(Rectify, numbirch::rectify)
  BIRCH_UNARY_GRAD(numbirch::rectify_grad)
  BIRCH_FORM
};

template<class Middle>
auto rectify(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::rectify(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Rectify);
  }
}

}
