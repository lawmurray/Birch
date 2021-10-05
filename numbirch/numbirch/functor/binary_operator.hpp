/**
 * @file
 */
#pragma once

#include "numbirch/functor/macro.hpp"
#include "numbirch/functor/function.hpp"

namespace numbirch {
template<class T>
struct add_functor {
  DEVICE T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<class T>
struct divide_functor {
  DEVICE T operator()(const T x, const T y) const {
    return x/y;
  }
};

template<class T, class U>
struct divide_scalar_functor {
  divide_scalar_functor(const U* a) :
      a(a) {
    //
  }
  DEVICE T operator()(const T x) const {
    return x/(*a);
  }
  const U* a;
};

template<class T>
struct equal_functor {
  DEVICE bool operator()(const T x, const T y) const {
    return x == y;
  }
};

template<class T>
struct greater_functor {
  DEVICE bool operator()(const T x, const T y) const {
    return x > y;
  }
};

template<class T>
struct greater_or_equal_functor {
  DEVICE bool operator()(const T x, const T y) const {
    return x >= y;
  }
};

template<class T>
struct less_functor {
  DEVICE bool operator()(const T x, const T y) const {
    return x < y;
  }
};

template<class T>
struct less_or_equal_functor {
  DEVICE bool operator()(const T x, const T y) const {
    return x <= y;
  }
};

struct logical_and_functor {
  DEVICE bool operator()(const bool x, const bool y) const {
    return x && y;
  }
};

struct logical_or_functor {
  DEVICE bool operator()(const bool x, const bool y) const {
    return x || y;
  }
};

template<class T>
struct multiply_functor {
  DEVICE T operator()(const T x, const T y) const {
    return x*y;
  }
};

template<class T, class U>
struct multiply_scalar_functor {
  multiply_scalar_functor(const U* a) :
      a(a) {
    //
  }
  DEVICE T operator()(const T x) const {
    return x*(*a);
  }
  const U* a;
};

template<class T>
struct not_equal_functor {
  DEVICE bool operator()(const T x, const T y) const {
    return x != y;
  }
};

template<class T>
struct subtract_functor {
  DEVICE T operator()(const T x, const T y) const {
    return x - y;
  }
};

}
