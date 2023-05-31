/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct IsFinite {
  BIRCH_UNARY_FORM(IsFinite, numbirch::isfinite)
  BIRCH_UNARY_GRAD(numbirch::isfinite_grad)
  BIRCH_FORM
};

template<class Middle>
auto isfinite(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::isfinite(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(IsFinite);
  }
}

}
