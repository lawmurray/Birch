/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<arithmetic To>
struct CastOp {
  BIRCH_TRANSFORM_EVAL(cast<To>)
  BIRCH_TRANSFORM_UNARY_GRAD(cast_grad<To>)
  BIRCH_TRANSFORM_SIZE
};

template<arithmetic To, argument T>
using Cast = Form<CastOp<To>,T>;

template<arithmetic To, argument T>
auto cast(T&& x) {
  return Cast<To,tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<arithmetic To, argument T>
auto cast(const Cast<To,T>& x) {
  return x;
}

}
