/**
 * @file
 * 
 * Functors used by multiple backends.
 */
#pragma once

#include "numbirch/function.hpp"
#include "numbirch/type.hpp"

namespace numbirch {

template<class T>
struct combine_functor {
  combine_functor(const T a , const T b, const T c, const T d) :
      a(a), b(b), c(c), d(d) {
    //
  }
  HOST_DEVICE T operator()(const T w, const T x, const T y, const T z) const {
    return a*w + b*x + c*y + d*z;
  }
  const T a, b, c, d;
};

template<class T>
struct combine4_functor {
  combine4_functor(const T a , const T b, const T c, const T d) :
      a(a), b(b), c(c), d(d) {
    //
  }
  HOST_DEVICE T operator()(const quad<T>& o) const {
    return a*o.first + b*o.second + c*o.third + d*o.fourth;
  }
  const T a, b, c, d;
};

}
