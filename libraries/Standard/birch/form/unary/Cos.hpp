/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Cos {
  BIRCH_UNARY_FORM(Cos, numbirch::cos)
  BIRCH_UNARY_GRAD(numbirch::cos_grad)
  BIRCH_FORM
};

template<class Middle>
auto cos(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::cos(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Cos);
  }
}

}
