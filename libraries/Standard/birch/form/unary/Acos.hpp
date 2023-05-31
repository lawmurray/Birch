/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Acos {
  BIRCH_UNARY_FORM(Acos, numbirch::acos)
  BIRCH_UNARY_GRAD(numbirch::acos_grad)
  BIRCH_FORM
};

template<class Middle>
auto acos(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::acos(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Acos);
  }
}

}
