/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::vec;
using numbirch::vec_grad;

template<class Middle>
struct Vec : public Unary<Middle> {
  template<class T>
  Vec(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(vec)
  BIRCH_UNARY_GRAD(vec_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Vec<Middle> vec(const Middle& m) {
  return Vec<Middle>(m);
}

}
