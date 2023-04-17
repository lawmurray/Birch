/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::simulate_wishart;

template<class Middle>
struct SimulateWishart : public Unary<Middle> {
  /**
   * Size.
   */
  Integer n;

  template<class T>
  SimulateWishart(T&& m, Integer n) :
      Unary<Middle>(std::forward<T>(m)),
      n(n) {
    //
  }

  BIRCH_UNARY_FORM(simulate_wishart, n)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
SimulateWishart<Middle> simulate_wishart(const Middle& m, const int n) {
  return SimulateWishart<Middle>(m, n);
}

}
