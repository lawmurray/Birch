/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Scal {
  BIRCH_UNARY_FORM(Scal)
};

BIRCH_UNARY_SIZE(Scal)
BIRCH_UNARY(Scal, numbirch::scal)
BIRCH_UNARY_GRAD(Scal, numbirch::scal_grad)

template<class Middle>
auto scal(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::scal(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Scal);
  }
}

}
