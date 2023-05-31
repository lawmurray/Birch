/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Tanh {
  BIRCH_UNARY_FORM(Tanh, numbirch::tanh)
  BIRCH_UNARY_GRAD(numbirch::tanh_grad)
  BIRCH_FORM
};

template<class Middle>
auto tanh(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::tanh(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Tanh);
  }
}

}
