/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Atan {
  BIRCH_UNARY_FORM(Atan)
};

BIRCH_UNARY_SIZE(Atan)
BIRCH_UNARY(Atan, numbirch::atan)
BIRCH_UNARY_GRAD(Atan, numbirch::atan_grad)

template<class Middle>
auto atan(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::atan(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Atan);
  }
}

}
