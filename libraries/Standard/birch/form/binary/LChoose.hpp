/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct LChoose {
  BIRCH_BINARY_FORM(LChoose, numbirch::lchoose)
  BIRCH_BINARY_GRAD(numbirch::lchoose_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto lchoose(const Left& l, const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> && numbirch::is_arithmetic_v<Right>) {
    return numbirch::lchoose(l, r);
  } else {
    return BIRCH_BINARY_CONSTRUCT(LChoose);
  }
}

}
