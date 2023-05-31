/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct LFact {
  BIRCH_UNARY_FORM(LFact, numbirch::lfact)
  BIRCH_UNARY_GRAD(numbirch::lfact_grad)
  BIRCH_FORM
};

template<class Middle>
auto lfact(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::lfact(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(LFact);
  }
}

}
