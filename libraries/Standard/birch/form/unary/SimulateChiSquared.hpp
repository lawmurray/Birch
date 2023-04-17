/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::simulate_chi_squared;

template<class Middle>
struct SimulateChiSquared : public Unary<Middle> {
  template<class T>
  SimulateChiSquared(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(simulate_chi_squared)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
SimulateChiSquared<Middle> simulate_chi_squared(const Middle& m) {
  return SimulateChiSquared<Middle>(m);
}

}
