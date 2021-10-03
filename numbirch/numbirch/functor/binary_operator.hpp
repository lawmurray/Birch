/**
 * @file
 */
#pragma once

#include "numbirch/functor/macro.hpp"

namespace numbirch {
template<class T>
struct add_functor {
  HOST_DEVICE T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<class T>
struct divide_functor {
  HOST_DEVICE T operator()(const T x, const T y) const {
    return x/y;
  }
};

template<class T, class U>
struct divide_scalar_functor {
  divide_scalar_functor(const U* a) :
      a(a) {
    //
  }
  HOST_DEVICE T operator()(const T x) const {
    return x/(*a);
  }
  const U* a;
};

template<class T>
struct multiply_functor {
  HOST_DEVICE T operator()(const T x, const T y) const {
    return x*y;
  }
};

template<class T, class U>
struct multiply_scalar_functor {
  multiply_scalar_functor(const U* a) :
      a(a) {
    //
  }
  HOST_DEVICE T operator()(const T x) const {
    return x*(*a);
  }
  const U* a;
};

template<class T>
struct subtract_functor {
  HOST_DEVICE T operator()(const T x, const T y) const {
    return x - y;
  }
};

}
