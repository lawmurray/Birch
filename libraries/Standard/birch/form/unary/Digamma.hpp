/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Digamma {
  BIRCH_UNARY_FORM(Digamma)
};

BIRCH_UNARY_SIZE(Digamma)
BIRCH_UNARY(Digamma, numbirch::digamma)

template<class Middle>
auto digamma(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::digamma(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Digamma);
  }
}

}
