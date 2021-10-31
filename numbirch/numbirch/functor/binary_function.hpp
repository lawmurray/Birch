/**
 * @file
 */
#pragma once

#include "numbirch/function.hpp"

namespace numbirch {

template<class T>
struct copysign_functor {
  HOST DEVICE T operator()(const T x, const T y) const {
    return copysign(x, y);
  }
};

template<class T>
struct digammap_functor {
  HOST DEVICE T operator()(const T x, const int y) const {
    return digamma(x, y);
  }
};

template<class T>
struct gamma_p_functor {
  HOST DEVICE T operator()(const T a, const T x) const {
    return gamma_p(a, x);
  }
};

template<class T>
struct gamma_q_functor {
  HOST DEVICE T operator()(const T a, const T x) const {
    return gamma_q(a, x);
  }
};

template<class T>
struct lbeta_functor {
  HOST DEVICE T operator()(const T x, const T y) const {
    return lbeta(x, y);
  }
};

template<class T>
struct lchoose_functor {
  HOST DEVICE T operator()(const int x, const int y) const {
    return lchoose<T>(x, y);
  }
};

template<class T>
struct lchoose_grad_functor {
  HOST DEVICE pair<T> operator()(const T d, const int x, const int y)
      const {
    return lchoose_grad<T>(d, x, y);
  }
};

template<class T>
struct lgammap_functor {
  HOST DEVICE T operator()(const T x, const T y) const {
    return lgamma(x, y);
  }
};

template<class T>
struct pow_functor {
  HOST DEVICE T operator()(const T x, const T y) const {
    return pow(x, y);
  }
};

template<class T>
struct single_functor {
  single_functor(const int* i, const int* j) :
      i(i), j(j) {
    //
  }
  HOST DEVICE T operator()(const int i, const int j) const {
    return (i == *this->i - 1 && j == *this->j - 1) ? T(1) : T(0);
  }
  const int *i, *j;
};

}
