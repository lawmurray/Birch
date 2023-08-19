/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct And {
  BIRCH_BINARY_FORM(And)
  BIRCH_BINARY_SIZE(And)
  BIRCH_BINARY_GRAD(And, logical_and_grad)

  /* bespoke implementation of evaluation functions, rather than using
   * BIRCH_BINARY_EVAL, to provide short circuit support */
  using value_t = decltype(numbirch::logical_and(birch::value(l), birch::value(r)));

  value_t value() const {
    constant();
    return eval();
  }
 
  value_t eval() const {
    if constexpr (std::same_as<Left,bool>) {
      if (birch::eval(l)) {
        return birch::eval(r);
      } else {
        return false;
      }
    } else {
      return numbirch::logical_and(birch::eval(l), birch::eval(r));
    }
  }
};

BIRCH_BINARY_TYPE(And)
BIRCH_BINARY_CALL(And, logical_and)

/* principle of least surprise: do not overload && as it will not short
 * circuit, leading to unintuitive errors */

}
