/**
 * @file
 */
#pragma once

#include "numbirch/function.hpp"

namespace numbirch {
template<class T, class U>
struct add_functor {
  HOST_DEVICE auto operator()(const T x, const U y) const {
    return x + y;
  }
};

template<class T, class U>
struct divide_functor {
  HOST_DEVICE auto operator()(const T x, const U y) const {
    return x/y;
  }
};

template<class T, class U>
struct divide_scalar_functor {
  divide_scalar_functor(const U* a) :
      a(a) {
    //
  }
  HOST_DEVICE auto operator()(const T x) const {
    return x/(*a);
  }
  const U* a;
};

template<class T, class U>
struct equal_functor {
  HOST_DEVICE bool operator()(const T x, const U y) const {
    return x == y;
  }
};

template<class T, class U>
struct greater_functor {
  HOST_DEVICE bool operator()(const T x, const U y) const {
    return x > y;
  }
};

template<class T, class U>
struct greater_or_equal_functor {
  HOST_DEVICE bool operator()(const T x, const U y) const {
    return x >= y;
  }
};

template<class T, class U>
struct less_functor {
  HOST_DEVICE bool operator()(const T x, const U y) const {
    return x < y;
  }
};

template<class T, class U>
struct less_or_equal_functor {
  HOST_DEVICE bool operator()(const T x, const U y) const {
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

template<class T, class U>
struct multiply_functor {
  HOST_DEVICE auto operator()(const T x, const U y) const {
    return x*y;
  }
};

template<class T, class U>
struct multiply_scalar_functor {
  multiply_scalar_functor(const U* a) :
      a(a) {
    //
  }
  HOST_DEVICE auto operator()(const T x) const {
    return x*(*a);
  }
  const U* a;
};

template<class T, class U>
struct not_equal_functor {
  HOST_DEVICE bool operator()(const T x, const U y) const {
    return x != y;
  }
};

template<class T, class U>
struct subtract_functor {
  HOST_DEVICE auto operator()(const T x, const U y) const {
    return x - y;
  }
};

}
