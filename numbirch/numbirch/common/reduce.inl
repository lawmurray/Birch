/**
 * @file
 */
#pragma once

#include "numbirch/reduce.hpp"
#include "numbirch/array.hpp"

namespace numbirch {

struct count_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE int operator()(const T x) const {
    return int((x == T(0)) ? 0 : 1);
  }
};

struct count_grad_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE real operator()(const T x) const {
    return real(0);
  }
};

struct sum_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return x;
  }
};

template<class T>
struct sum_grad_functor {
  T g;
  sum_grad_functor(const T& g) : g(g) {
    //
  }
  template<class U>
  NUMBIRCH_HOST_DEVICE auto operator()(const U x) const {
    return get(g);
  }
};

template<class R, class T, class>
real_t<T> count_grad(const Array<real,0>& g, const Array<R,0>& y,
    const T& x) {
  prefetch(x);
  return transform(x, count_grad_functor());
}

template<class R, class T, class>
real_t<T> sum_grad(const Array<real,0>& g, const Array<R,0>& y,
    const T& x) {
  prefetch(x);
  return transform(x, sum_grad_functor(sliced(g)));
}

}
