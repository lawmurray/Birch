/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Erf {
  BIRCH_UNARY_FORM(Erf, numbirch::erf)
  BIRCH_UNARY_GRAD(numbirch::erf_grad)
  BIRCH_FORM
};

template<class Middle>
auto erf(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::erf(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Erf);
  }
}

}
