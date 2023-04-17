/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::simulate_exponential;

template<class Middle>
struct SimulateExponential : public Unary<Middle> {
  template<class T>
  SimulateExponential(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(simulate_exponential)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
SimulateExponential<Middle> simulate_exponential(const Middle& m) {
  return SimulateExponential<Middle>(m);
}

}
