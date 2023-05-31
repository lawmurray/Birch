/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct IsInf {
  BIRCH_UNARY_FORM(IsInf, numbirch::isinf)
  BIRCH_UNARY_GRAD(numbirch::isinf_grad)
  BIRCH_FORM
};

template<class Middle>
auto isinf(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::isinf(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(IsInf);
  }
}

}
