/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Floor {
  BIRCH_UNARY_FORM(Floor, numbirch::floor)
  BIRCH_UNARY_GRAD(numbirch::floor_grad)
  BIRCH_FORM
};

template<class Middle>
auto floor(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::floor(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Floor);
  }
}

}
