/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Expm1 {
  BIRCH_UNARY_FORM(Expm1, numbirch::expm1)
  BIRCH_UNARY_GRAD(numbirch::expm1_grad)
  BIRCH_FORM
};

template<class Middle>
auto expm1(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::expm1(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Expm1);
  }
}

}
