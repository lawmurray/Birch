/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {
using numbirch::log;
using numbirch::log_grad;

template<class Middle>
struct Log : public Unary<Middle> {
  template<class T>
  Log(T&& m) :
      Unary<Middle>(std::forward<T>(m)) {
    //
  }

  BIRCH_UNARY_FORM(log)
  BIRCH_UNARY_GRAD(log_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

template<class Middle, std::enable_if_t<is_delay_v<Middle>,int> = 0>
Log<Middle> log(const Middle& m) {
  return Log<Middle>(m);
}

}
