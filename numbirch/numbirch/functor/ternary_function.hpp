/**
 * @file
 */
#pragma once

#include "numbirch/function.hpp"
#include "numbirch/type.hpp"

namespace numbirch {

template<class T>
struct ibeta_functor {
  HOST_DEVICE T operator()(const T a, const T b, const T x) const {
    return ibeta(a, b, x);
  }
};

template<class T>
struct if_then_else_functor {
  HOST_DEVICE T operator()(const bool x, const T y, const T z) const {
    return x ? y : z;
  }
};

}
