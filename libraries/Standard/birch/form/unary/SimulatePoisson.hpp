/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::simulate_poisson;

template<class Middle>
struct SimulatePoisson : public Unary<Middle> {
  template<class T>
  SimulatePoisson(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(simulate_poisson)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
SimulatePoisson<Middle> simulate_poisson(const Middle& m) {
  return SimulatePoisson<Middle>(m);
}

}
