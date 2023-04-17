/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {
using numbirch::simulate_uniform_int;

template<class Left, class Right>
struct SimulateUniformInt : public Binary<Left,Right> {
  template<class T, class U>
  SimulateUniformInt(T&& l, U&& r) :
      Binary<Left,Right>(std::forward<T>(l), std::forward<U>(r)) {
    //
  }

  BIRCH_BINARY_FORM(simulate_uniform_int)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Right, std::enable_if_t<
    is_delay_v<Left,Right>,int> = 0>
SimulateUniformInt<Left,Right> simulate_uniform_int(const Left& l, const Right& r) {
  return SimulateUniformInt<Left,Right>(l, r);
}

}
