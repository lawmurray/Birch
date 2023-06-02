/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Sinh {
  BIRCH_UNARY_FORM(Sinh)
};

BIRCH_UNARY_SIZE(Sinh)
BIRCH_UNARY(Sinh, numbirch::sinh)
BIRCH_UNARY_GRAD(Sinh, numbirch::sinh_grad)

template<class Middle>
auto sinh(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::sinh(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Sinh);
  }
}

}
