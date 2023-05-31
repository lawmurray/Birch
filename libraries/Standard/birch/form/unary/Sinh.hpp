/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Sinh {
  BIRCH_UNARY_FORM(Sinh, numbirch::sinh)
  BIRCH_UNARY_GRAD(numbirch::sinh_grad)
  BIRCH_FORM
};

template<class Middle>
auto sinh(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::sinh(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Sinh);
  }
}

}
