/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Log1p {
  BIRCH_UNARY_FORM(Log1p, numbirch::log1p)
  BIRCH_UNARY_GRAD(numbirch::log1p_grad)
  BIRCH_FORM
};

template<class Middle>
auto log1p(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::log1p(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Log1p);
  }
}

}
