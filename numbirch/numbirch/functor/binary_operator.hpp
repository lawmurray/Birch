/**
 * @file
 */
#pragma once

#include "numbirch/function.hpp"

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

template<class T>
struct divide_scalar_functor {
  divide_scalar_functor(const T* a) :
      a(a) {
    //
  }
  HOST_DEVICE T operator()(const T x) const {
    return x/(*a);
  }
  const T* a;
};

template<class T>
struct equal_functor {
  HOST_DEVICE bool operator()(const T x, const T y) const {
    return x == y;
  }
};

template<class T>
struct greater_functor {
  HOST_DEVICE bool operator()(const T x, const T y) const {
    return x > y;
  }
};

template<class T>
struct greater_or_equal_functor {
  HOST_DEVICE bool operator()(const T x, const T y) const {
    return x >= y;
  }
};

template<class T>
struct less_functor {
  HOST_DEVICE bool operator()(const T x, const T y) const {
    return x < y;
  }
};

template<class T>
struct less_or_equal_functor {
  HOST_DEVICE bool operator()(const T x, const T y) const {
    return x <= y;
  }
};

struct logical_and_functor {
  HOST_DEVICE bool operator()(const bool x, const bool y) const {
    return x && y;
  }
};

struct logical_or_functor {
  HOST_DEVICE bool operator()(const bool x, const bool y) const {
    return x || y;
  }
};

template<class T>
struct multiply_functor {
  HOST_DEVICE auto operator()(const T x, const T y) const {
    return x*y;
  }
};

template<class T>
struct multiply_scalar_functor {
  multiply_scalar_functor(const T* a) :
      a(a) {
    //
  }
  HOST_DEVICE T operator()(const T x) const {
    return x*(*a);
  }
  const T* a;
};

template<class T>
struct not_equal_functor {
  HOST_DEVICE bool operator()(const T x, const T y) const {
    return x != y;
  }
};

template<class T>
struct subtract_functor {
  HOST_DEVICE auto operator()(const T x, const T y) const {
    return x - y;
  }
};

}
