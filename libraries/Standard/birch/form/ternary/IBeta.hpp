/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<class Left, class Middle, class Right>
struct IBeta {
  BIRCH_TERNARY_FORM(IBeta, numbirch::ibeta)
  BIRCH_FORM
};

template<class Left, class Middle, class Right>
auto ibeta(const Left& l, const Middle& m, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> &&
      numbirch::is_arithmetic_v<Middle> &&
      numbirch::is_arithmetic_v<Right>) {
    return numbirch::ibeta(l, m, r);
  } else {
    return BIRCH_TERNARY_CONSTRUCT(IBeta);
  }
}

}
