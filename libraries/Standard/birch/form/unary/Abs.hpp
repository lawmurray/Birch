/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct AbsOp {
  BIRCH_TRANSFORM_EVAL(abs)
  BIRCH_TRANSFORM_UNARY_GRAD(abs_grad)
  BIRCH_TRANSFORM_SIZE
};

template<argument T>
using Abs = Form<AbsOp,T>;

template<argument T>
auto abs(T&& x) {
  return Abs<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T>
auto abs(const Abs<T>& x) {
  return x;
}

}
