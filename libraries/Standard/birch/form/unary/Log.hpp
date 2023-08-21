/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct LogOp {
  BIRCH_TRANSFORM_EVAL(log)
  BIRCH_TRANSFORM_UNARY_GRAD(log_grad)
  BIRCH_TRANSFORM_SIZE
};

template<argument T>
using Log = Form<LogOp,T>;

template<argument T>
auto log(T&& x) {
  return Log<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T>
decltype(auto) log(const Exp<T>& x) {
  return std::get<0>(x.tup);
}

}
