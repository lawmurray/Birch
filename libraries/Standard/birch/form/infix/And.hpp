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
    if constexpr (std::same_as<Left,bool>) {
      if (birch::value(l)) {
        return birch::value(r);
      } else {
        return false;
      }
    } else {
      return numbirch::logical_and(birch::value(l), birch::value(r));
    }
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
 
  value_t peek() const {
    if constexpr (std::same_as<Left,bool>) {
      if (birch::peek(l)) {
        return birch::peek(r);
      } else {
        return false;
      }
    } else {
      return numbirch::logical_and(birch::peek(l), birch::peek(r));
    }
  }
 
  value_t move(const MoveVisitor& visitor) const {
    if constexpr (std::same_as<Left,bool>) {
      if (birch::move(l, visitor)) {
        return birch::move(r, visitor);
      } else {
        return false;
      }
    } else {
      return numbirch::logical_and(birch::move(l, visitor),
          birch::move(r, visitor));
    }
  }
};

BIRCH_BINARY_TYPE(And)
BIRCH_BINARY_CALL(And, logical_and)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator&&(Left&& l, Right&& r) {
  return logical_and(std::forward<Left>(l), std::forward<Right>(r));
}

}
