/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct IsInf {
  BIRCH_UNARY_FORM(IsInf)
};

BIRCH_UNARY_SIZE(IsInf)
BIRCH_UNARY(IsInf, numbirch::isinf)
BIRCH_UNARY_GRAD(IsInf, numbirch::isinf_grad)

template<class Middle>
auto isinf(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::isinf(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(IsInf);
  }
}

}
