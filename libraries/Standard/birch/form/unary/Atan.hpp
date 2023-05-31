/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Atan {
  BIRCH_UNARY_FORM(Atan, numbirch::atan)
  BIRCH_UNARY_GRAD(numbirch::atan_grad)
  BIRCH_FORM
};

template<class Middle>
auto atan(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::atan(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Atan);
  }
}

}
