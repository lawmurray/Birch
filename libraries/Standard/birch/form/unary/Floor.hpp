/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Floor {
  BIRCH_UNARY_FORM(Floor)
};

BIRCH_UNARY_SIZE(Floor)
BIRCH_UNARY(Floor, numbirch::floor)
BIRCH_UNARY_GRAD(Floor, numbirch::floor_grad)

template<class Middle>
auto floor(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::floor(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Floor);
  }
}

}
