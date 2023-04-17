/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {
using numbirch::where;
using numbirch::where_grad1;
using numbirch::where_grad2;
using numbirch::where_grad3;

template<class Left, class Middle, class Right>
struct Where : public Ternary<Left,Middle,Right> {
  template<class T, class U, class V>
  Where(T&& l, U&& m, V&& r) :
      Ternary<Left,Middle,Right>(std::forward<T>(l), std::forward<U>(m),
      std::forward<V>(r)) {
    //
  }

  BIRCH_TERNARY_FORM(where)
  BIRCH_TERNARY_GRAD(where_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Left, class Middle, class Right, std::enable_if_t<
    is_delay_v<Left,Middle,Right>,int> = 0>
Where<Left,Middle,Right> where(const Left& l, const Middle& m,
    const Right& r) {
  return Where<Left,Middle,Right>(l, m, r);
}

}
