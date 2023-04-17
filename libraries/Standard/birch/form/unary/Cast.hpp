/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::cast;
using numbirch::cast_grad;

template<class To, class Middle>
struct Cast : public Unary<Middle> {
  template<class T>
  Cast(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(cast<To>)
  BIRCH_UNARY_GRAD(cast_grad<To>)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class To, class Middle, std::enable_if_t<
    is_delay_v<Middle>,int> = 0>
Cast<To,Middle> cast(const Middle& m) {
  return Cast<To,Middle>(m);
}

}
