/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Asin {
  BIRCH_UNARY_FORM(Asin)
};

BIRCH_UNARY_SIZE(Asin)
BIRCH_UNARY(Asin, numbirch::asin)
BIRCH_UNARY_GRAD(Asin, numbirch::asin_grad)

template<class Middle>
auto asin(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::asin(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Asin);
  }
}

}
