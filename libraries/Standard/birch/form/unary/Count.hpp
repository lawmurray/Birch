/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Count {
  BIRCH_UNARY_FORM(Count)
};

BIRCH_UNARY_SIZE(Count)
BIRCH_UNARY(Count, numbirch::count)
BIRCH_UNARY_GRAD(Count, numbirch::count_grad)

template<class Middle>
auto count(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::count(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Count);
  }
}

}
