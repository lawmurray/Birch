/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::simulate_uniform;

template<class Left, class Right>
struct SimulateUniform : public Binary<Left,Right> {
  template<class T, class U>
  SimulateUniform(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(simulate_uniform)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
SimulateUniform<Left,Right> simulate_uniform(const Left& l, const Right& r) {
  return SimulateUniform<Left,Right>(l, r);
}

}
