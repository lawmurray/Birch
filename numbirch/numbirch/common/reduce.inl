/**
 * @file
 */
#pragma once

#include "numbirch/reduce.hpp"
#include "numbirch/transform.hpp"
#include "numbirch/array.hpp"

namespace numbirch {

struct count_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE int operator()(const T x) const {
    return x == T(0) ? 0 : 1;
  }
};

struct count_grad_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x) const {
    return (x != T(0)) ? g : real(0);
  }
};

struct sum_grad_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const T x) const {
    return g;
  }
};

struct min_grad_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const T y, const U x)
      const {
    return (x == y) ? g : real(0);
  }
};

struct max_grad_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const T y, const U x)
      const {
    return (x == y) ? g : real(0);
  }
};

template<class T, class>
real_t<T> count_grad(const Array<real,0>& g, const Array<int,0>& y,
    const T& x) {
  return transform(g, x, count_grad_functor());
}

template<class T, class>
real_t<T> sum_grad(const Array<real,0>& g, const Array<value_t<T>,0>& y,
    const T& x) {
  return transform(g, x, sum_grad_functor());
}

template<class T, class>
real_t<T> min_grad(const Array<real,0>& g, const Array<value_t<T>,0>& y,
    const T& x) {
  return transform(g, y, x, min_grad_functor());
}

template<class T, class>
real_t<T> max_grad(const Array<real,0>& g, const Array<value_t<T>,0>& y,
    const T& x) {
  return transform(g, y, x, max_grad_functor());
}

template<class T, class>
real_t<T> cumsum_grad(const real_t<T>& g, const T& y, const T& x) {
  return sub(sum(g), sub(cumsum(g), g));
}

}
