/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct SimulateBernoulli {
  BIRCH_UNARY_FORM(SimulateBernoulli, numbirch::simulate_bernoulli)
  BIRCH_FORM
};

template<class Middle>
auto simulate_bernoulli(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::simulate_bernoulli(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(SimulateBernoulli);
  }
}

}
