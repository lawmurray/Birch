/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::sum;
using numbirch::sum_grad;

template<class Middle>
struct Sum : public Unary<Middle> {
  template<class T>
  Sum(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(sum)
  BIRCH_UNARY_GRAD(sum_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Sum<Middle> sum(const Middle& m) {
  return Sum<Middle>(m);
}

}
