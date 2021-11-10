/**
 * @file
 */
#pragma once

#include "numbirch/function.hpp"
#include "numbirch/type.hpp"

namespace numbirch {
struct negate_functor {
  template<class T>
  HOST DEVICE T operator()(const T x) const {
    return -x;
  }
};

struct add_functor {
  template<class T, class U>
  HOST DEVICE promote_t<T,U> operator()(const T x, const U y) const {
    using V = promote_t<T,U>;
    return V(x) + V(y);
  }
};

struct subtract_functor {
  template<class T, class U>
  HOST DEVICE promote_t<T,U> operator()(const T x, const U y)
      const {
    using V = promote_t<T,U>;
    return V(x) - V(y);
  }
};

struct multiply_functor {
  template<class T, class U>
  HOST DEVICE promote_t<T,U> operator()(const T x, const U y) const {
    using V = promote_t<T,U>;
    return V(x)*V(y);
  }
};

struct divide_functor {
  template<class T, class U>
  HOST DEVICE promote_t<T,U> operator()(const T x, const U y) const {
    using V = promote_t<T,U>;
    return V(x)/V(y);
  }
};

struct not_functor {
  HOST DEVICE bool operator()(const bool x) const {
    return !x;
  }
};

struct and_functor {
  HOST DEVICE bool operator()(const bool x, const bool y) const {
    return x && y;
  }
};

struct or_functor {
  HOST DEVICE bool operator()(const bool x, const bool y) const {
    return x || y;
  }
};

struct equal_functor {
  template<class T, class U>
  HOST DEVICE bool operator()(const T x, const U y) const {
    using V = promote_t<T,U>;
    return V(x) == V(y);
  }
};

struct not_equal_functor {
  template<class T, class U>
  HOST DEVICE bool operator()(const T x, const U y) const {
    using V = promote_t<T,U>;
    return V(x) != V(y);
  }
};

struct less_functor {
  template<class T, class U>
  HOST DEVICE bool operator()(const T x, const U y) const {
    using V = promote_t<T,U>;
    return V(x) < V(y);
  }
};

struct less_or_equal_functor {
  template<class T, class U>
  HOST DEVICE bool operator()(const T x, const U y) const {
    using V = promote_t<T,U>;
    return V(x) <= V(y);
  }
};

struct greater_functor {
  template<class T, class U>
  HOST DEVICE bool operator()(const T x, const U y) const {
    using V = promote_t<T,U>;
    return V(x) > V(y);
  }
};

struct greater_or_equal_functor {
  template<class T, class U>
  HOST DEVICE bool operator()(const T x, const U y) const {
    using V = promote_t<T,U>;
    return V(x) >= V(y);
  }
};

struct abs_functor {
  template<class T>
  HOST DEVICE T operator()(const T x) const {
    return std::abs(x);
  }
};

template<class T>
struct acos_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::acos(x);
  }
};

template<class T>
struct asin_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::asin(x);
  }
};

template<class T>
struct atan_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::atan(x);
  }
};

struct ceil_functor {
  template<class T>
  HOST DEVICE T operator()(const T x) const {
    return ceil(x);
  }
};

template<class T>
struct cos_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::cos(x);
  }
};

template<class T>
struct cosh_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::cosh(x);
  }
};

struct count_functor {
  template<class T>
  HOST DEVICE int operator()(const T x) const {
    return (x == T(0)) ? 0 : 1;
  }
};

template<class T>
struct diagonal_functor {
  diagonal_functor(const T a) :
      a(a) {
    //
  }
  HOST DEVICE auto operator()(const int i, const int j) const {
    return (i == j) ? element(a) : 0;
  }
  const T a;
};

template<class T>
struct digamma_functor {
  HOST DEVICE T operator()(const T x) const {
    return digamma(x);
  }
  HOST DEVICE T operator()(const T x, const int y) const {
    return digamma(x, y);
  }
};

template<class T>
struct exp_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::exp(x);
  }
};

template<class T>
struct expm1_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::expm1(x);
  }
};

struct floor_functor {
  template<class T>
  HOST DEVICE T operator()(const T x) const {
    return floor(x);
  }
};

template<class T>
struct lfact_functor {
  HOST DEVICE T operator()(const T x) const {
    return lfact(x);
  }
};

template<class T>
struct lfact_grad_functor {
  HOST DEVICE T operator()(const T g, const T x) const {
    return lfact_grad(g, x);
  }
};

template<class T>
struct lgamma_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::lgamma(x);
  }
  HOST DEVICE T operator()(const T x, const T y) const {
    return lgamma(x, y);
  }
};

template<class T>
struct log_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::log(x);
  }
};

template<class T>
struct log1p_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::log1p(x);
  }
};

template<class T>
struct log_abs_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::log(std::abs(x));
  }
};

template<class T>
struct log_square_functor {
  HOST DEVICE T operator()(const T x) const {
    return 2.0*std::log(x);
  }
};

template<class T>
struct rcp_functor {
  HOST DEVICE T operator()(const T x) const {
    return rcp(x);
  }
};

template<class T>
struct rectify_functor {
  HOST DEVICE T operator()(const T x) const {
    return rectify(x);
  }
};

template<class T>
struct rectify_grad_functor {
  HOST DEVICE T operator()(const T g, const T x) const {
    return rectify_grad(g, x);
  }
};

struct round_functor {
  template<class T>
  HOST DEVICE T operator()(const T x) const {
    return round(x);
  }
};

template<class T>
struct sin_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::sin(x);
  }
};

template<class T>
struct sinh_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::sinh(x);
  }
};

template<class T>
struct sqrt_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::sqrt(x);
  }
};

template<class T>
struct tan_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::tan(x);
  }
};

template<class T>
struct tanh_functor {
  HOST DEVICE T operator()(const T x) const {
    return std::tanh(x);
  }
};

struct copysign_functor {
  template<class T, class U>
  HOST DEVICE T operator()(const T x, const U y) const {
    return copysign(x, y);
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
  HOST DEVICE pair<T,T> operator()(const T d, const int x, const int y)
      const {
    return lchoose_grad<T,T>(d, x, y);
  }
};

template<class T>
struct pow_functor {
  HOST DEVICE T operator()(const T x, const T y) const {
    return pow(x, y);
  }
};

template<class T, class U, class V = int>
struct single_functor {
  single_functor(const U k, const V l = 1) :
      k(k), l(l) {
    //
  }
  HOST DEVICE T operator()(const int i, const int j) const {
    return (i == element(k) - 1 && j == element(l) - 1) ? T(1) : T(0);
  }
  const U k;
  const V l;
};

template<class T>
struct ibeta_functor {
  HOST DEVICE T operator()(const T a, const T b, const T x) const {
    return ibeta(a, b, x);
  }
};

template<class T>
struct if_then_else_functor {
  HOST DEVICE T operator()(const bool x, const T y, const T z) const {
    return x ? y : z;
  }
};

}
