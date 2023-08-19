/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<argument Left, argument Middle, argument Right>
struct Where {
  BIRCH_TERNARY_FORM(Where)
  BIRCH_TERNARY_SIZE(Where)
  BIRCH_TERNARY_GRAD(Where, where_grad)

  /* bespoke implementation of evaluation functions, rather than using
   * BIRCH_TERNARY_EVAL, to provide short circuit support */
  using value_t = decltype(numbirch::where(birch::value(l), birch::value(m),
      birch::value(r)));

  value_t value() const {
    constant();
    return eval();
  }
 
  value_t eval() const {
    if constexpr (std::same_as<Left,bool>) {
      if (birch::eval(l)) {
        return birch::eval(m);
      } else {
        return birch::eval(r);
      }
    } else {
      return numbirch::where(birch::eval(l), birch::eval(m), birch::eval(r));
    }
  }
};

BIRCH_TERNARY_TYPE(Where)
BIRCH_TERNARY_CALL(Where, where)

}
