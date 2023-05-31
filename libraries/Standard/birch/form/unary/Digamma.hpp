/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Digamma {
  BIRCH_UNARY_FORM(Digamma, numbirch::digamma)
  BIRCH_NO_GRAD
  BIRCH_FORM
};

template<class Middle>
auto digamma(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::digamma(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Digamma);
  }
}

}
