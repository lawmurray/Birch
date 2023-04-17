/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::simulate_bernoulli;

template<class Middle>
struct SimulateBernoulli : public Unary<Middle> {
  template<class T>
  SimulateBernoulli(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(simulate_bernoulli)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
SimulateBernoulli<Middle> simulate_bernoulli(const Middle& m) {
  return SimulateBernoulli<Middle>(m);
}

}
