/**
 * @file
 * 
 * Functors used by multiple backends.
 */
#pragma once

#include "numbirch/function.hpp"

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
  HOST_DEVICE T operator()(const std::tuple<T,T,T,T>& o) const {
    return a*std::get<0>(o) + b*std::get<1>(o) + c*std::get<2>(o) +
        d*std::get<3>(o);
  }
  const T a, b, c, d;
};

template<class T>
struct diagonal_functor {
  diagonal_functor(const T* a) :
      a(a) {
    //
  }
  HOST_DEVICE T operator()(const int i, const int j) const {
    return (i == j) ? *a : T(0);
  }
  const T* a;
};

}
