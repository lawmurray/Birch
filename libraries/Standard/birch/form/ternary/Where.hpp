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
    if constexpr (std::same_as<Left,bool>) {
      if (birch::value(l)) {
        return birch::value(m);
      } else {
        return birch::value(r);
      }
    } else {
      return numbirch::where(birch::value(l), birch::value(m), birch::value(r));
    }
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
 
  value_t peek() const {
    if constexpr (std::same_as<Left,bool>) {
      if (birch::peek(l)) {
        return birch::peek(m);
      } else {
        return birch::peek(r);
      }
    } else {
      return numbirch::where(birch::peek(l), birch::peek(m), birch::peek(r));
    }
  }
 
  value_t move(const MoveVisitor& visitor) const {
    if constexpr (std::same_as<Left,bool>) {
      if (birch::move(l, visitor)) {
        return birch::move(m, visitor);
      } else {
        return birch::move(r, visitor);
      }
    } else {
      return numbirch::where(birch::move(l, visitor), birch::move(m, visitor),
          birch::move(r, visitor));
    }
  }

};

BIRCH_TERNARY_TYPE(Where)
BIRCH_TERNARY_CALL(Where, where)

}
