/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Or {
  BIRCH_BINARY_FORM(Or)
  BIRCH_BINARY_SIZE(Or)
  BIRCH_BINARY_GRAD(Or, logical_or_grad)

  /* bespoke implementation of evaluation functions, rather than using
   * BIRCH_BINARY_EVAL, to provide short circuit support */
  using value_t = decltype(numbirch::logical_or(birch::value(l), birch::value(r)));

  value_t value() const {
    if constexpr (std::same_as<Left,bool>) {
      if (birch::value(l)) {
        return true;
      } else {
        return birch::value(r);
      }
    } else {
      return numbirch::logical_or(birch::value(l), birch::value(r));
    }
  }
 
  value_t eval() const {
    if constexpr (std::same_as<Left,bool>) {
      if (birch::eval(l)) {
        return true;
      } else {
        return birch::eval(r);
      }
    } else {
      return numbirch::logical_or(birch::eval(l), birch::eval(r));
    }
  }
 
  value_t peek() const {
    if constexpr (std::same_as<Left,bool>) {
      if (birch::peek(l)) {
        return true;
      } else {
        return birch::peek(r);
      }
    } else {
      return numbirch::logical_or(birch::peek(l), birch::peek(r));
    }
  }
 
  value_t move(const MoveVisitor& visitor) const {
    if constexpr (std::same_as<Left,bool>) {
      if (birch::move(l, visitor)) {
        return true;
      } else {
        return birch::move(r, visitor);
      }
    } else {
      return numbirch::logical_or(birch::move(l, visitor),
          birch::move(r, visitor));
    }
  }
};

BIRCH_BINARY_TYPE(Or)
BIRCH_BINARY_CALL(Or, logical_or)

/* principle of least surprise: do not overload || as it will not short
 * circuit, leading to unintuitive errors */

}
