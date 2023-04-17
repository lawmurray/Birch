/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::count;
using numbirch::count_grad;

template<class Middle>
struct Count : public Unary<Middle> {
  template<class T>
  Count(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(count)
  BIRCH_UNARY_GRAD(count_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Count<Middle> count(const Middle& m) {
  return Count<Middle>(m);
}

}
