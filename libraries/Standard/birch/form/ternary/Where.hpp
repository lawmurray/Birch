/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<class Left, class Middle, class Right>
struct Where {
  BIRCH_TERNARY_FORM(Where)
};

BIRCH_TERNARY_SIZE(Where)
BIRCH_TERNARY(Where, numbirch::where)
BIRCH_TERNARY_GRAD(Where, numbirch::where_grad)

template<class Left, class Middle, class Right>
auto where(const Left& l, const Middle& m, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> &&
      numbirch::is_arithmetic_v<Middle> &&
      numbirch::is_arithmetic_v<Right>) {
    return numbirch::where(l, m, r);
  } else {
    return BIRCH_TERNARY_CONSTRUCT(Where);
  }
}

}
