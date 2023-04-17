/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::floor;
using numbirch::floor_grad;

template<class Middle>
struct Floor : public Unary<Middle> {
  template<class T>
  Floor(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(floor)
  BIRCH_UNARY_GRAD(floor_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Floor<Middle> floor(const Middle& m) {
  return Floor<Middle>(m);
}

}
