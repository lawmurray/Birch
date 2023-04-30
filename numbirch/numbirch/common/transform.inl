/**
 * @file
 */
#pragma once

#include "numbirch/transform.hpp"

#if defined(HAVE_UNSUPPORTED_EIGEN_SPECIALFUNCTIONS)
#include <unsupported/Eigen/SpecialFunctions>
#elif defined(HAVE_EIGEN3_UNSUPPORTED_EIGEN_SPECIALFUNCTIONS)
#include <eigen3/unsupported/Eigen/SpecialFunctions>
#endif

namespace numbirch {
/**
 * @internal
 * 
 * Performs the inverse operation of a scalar broadcast during gradient
 * computation. That is, if a scalar was broadcast during the forward pass,
 * upstream gradients must be aggregated, by summation, during the backward
 * pass.
 */
template<int D, class T>
constexpr auto aggregate(const T& x) {
  if constexpr (D == 0 && dimension_v<T> != 0) {
    return sum(x);
  } else {
    return x;
  }
}

template<class R>
struct cast_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE R operator()(const T x) const {
    return R(x);
  }
};

struct neg_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return -x;
  }
};

struct add_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE promote_t<T,U> operator()(const T x, const U y) const {
    return x + y;
  }
};

struct sub_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE promote_t<T,U> operator()(const T x, const U y) const {
    return x - y;
  }
};

struct not_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x) const {
    return bool(!x);
  }
};

struct and_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x && y);
  }
};

struct or_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x || y);
  }
};

struct equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x == y);
  }
};

struct not_equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x != y);
  }
};

struct less_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x < y);
  }
};

struct less_or_equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x <= y);
  }
};

struct greater_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x > y);
  }
};

struct greater_or_equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x >= y);
  }
};

struct abs_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::abs(x);
  }
};

struct abs_grad_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x) const {
    return std::copysign(g, x);
  }
};

struct acos_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::acos(x);
  }
};

struct acos_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return -g/std::sqrt(real(1.0) - x*x);
  }
};

struct asin_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::asin(x);
  }
};

struct asin_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return g/std::sqrt(real(1) - x*x);
  }
};

struct atan_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::atan(x);
  }
};

struct atan_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return g/(real(1) + x*x);
  }
};

struct ceil_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    if constexpr (is_int_v<T> || is_bool_v<T>) {
      return x;
    } else {
      return std::ceil(x);
    }
  }
};

struct cos_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::cos(x);
  }
};

struct cos_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return -g*std::sin(x);
  }
};

struct cosh_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::cosh(x);
  }
};

struct cosh_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return -g*std::sinh(x);
  }
};

struct digamma_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return Eigen::numext::digamma(x);
  }
  NUMBIRCH_HOST_DEVICE real operator()(const real x, const int y) const {
    real z = 0;
    for (int i = 1; i <= y; ++i) {
      z += Eigen::numext::digamma(x + real(0.5)*(1 - i));
    }
    return z;
  }
};

struct div_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE promote_t<T,U> operator()(const T x, const U y) const {
    return x/y;
  }
};

struct div_grad1_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x, const U y)
      const {
    return g/y;
  }
};

struct div_grad2_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x, const U y)
      const {
    return -g*x/(y*y);
  }
};

struct exp_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::exp(x);
  }
};

struct expm1_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::expm1(x);
  }
};

struct floor_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    if constexpr (is_int_v<T> || is_bool_v<T>) {
      return x;
    } else {
      return std::floor(x);
    }
  }
};

struct hadamard_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE promote_t<T,U> operator()(const T x, const U y) const {
    if constexpr (is_bool_v<T> && is_bool_v<U>) {
      return x && y;  // avoids compiler warning
    } else {
      return x*y;
    }
  }
};

struct hadamard_grad1_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x, const U y)
      const {
    return g*y;
  }
};

struct hadamard_grad2_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x, const U y)
      const {
    return g*x;
  }
};

struct isfinite_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x) const {
    return std::isfinite(x);
  }
};

struct isinf_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x) const {
    return std::isinf(x);
  }
};

struct isnan_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x) const {
    return std::isnan(x);
  }
};

struct lfact_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::lgamma(x + real(1));
  }
};

struct lfact_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return g*Eigen::numext::digamma(x + real(1));
  }
};

struct lgamma_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::lgamma(x);
  }
  NUMBIRCH_HOST_DEVICE real operator()(const real x, const real y) const {
    real z = real(0.25)*y*(y - 1)*std::log(real(PI));
    for (int i = 1; i <= y; ++i) {
      z += std::lgamma(x + real(0.5)*(1 - i));
    }
    return z;
  }
};

struct lgamma_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return g*Eigen::numext::digamma(x);
  }
};

struct lgamma_grad1_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x,
      const int y) const {
    real z = real(0);
    for (int i = 1; i <= y; ++i) {
      z += Eigen::numext::digamma(x + real(0.5)*(1 - i));
    }
    return g*z;
  }
};

struct lgamma_grad2_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x,
      const int y) const {
    return real(0);
  }
};

struct log_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::log(x);
  }
};

struct log_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return g/x;
  }
};

struct log1p_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::log1p(x);
  }
};

struct log1p_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return g/(x + real(1));
  }
};

struct log_abs_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::log(std::abs(x));
  }
};

struct log_square_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return real(2)*std::log(x);
  }
};

struct rectify_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    /* this is written to ensure that NaN propagates, i.e. if isnan(x)
     * then the condition is false and NaN is returned */
    if (x <= T(0)) {
      return T(0);
    } else {
      return x;
    }
  }
};

struct rectify_grad_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x) const {
    return (x <= T(0)) ? real(0) : g;
  }
};

struct round_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    if constexpr (is_int_v<T> || is_bool_v<T>) {
      return x;
    } else {
      return std::round(x);
    }
  }
};

struct sin_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::sin(x);
  }
};

struct sin_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return g*std::cos(x);
  }
};

struct sinh_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::sinh(x);
  }
};

struct sinh_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return g*std::cosh(x);
  }
};

struct sqrt_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::sqrt(x);
  }
};

struct sqrt_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return g*real(0.5)/std::sqrt(x);
  }
};

struct tan_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::tan(x);
  }
};

struct tan_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return g*(real(1) + std::pow(std::tan(x), real(2)));
  }
};

struct tanh_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x) const {
    return std::tanh(x);
  }
};

struct tanh_grad_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x) const {
    return g*(real(1) + std::pow(std::tanh(x), real(2)));
  }
};

struct copysign_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE promote_t<T,U> operator()(const T x, const U y) const {
    if constexpr (is_int_v<T> || is_bool_v<T>) {
      /* don't use std::copysign, as it promotes to floating point, which
       * we don't wish to do here */
      return (y >= U(0)) ? std::abs(x) : -std::abs(x);
    } else {
      return std::copysign(real(x), real(y));
    }
  }
};

struct copysign_grad1_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x, const U y)
      const {
    T z;
    if constexpr (is_int_v<T> || is_bool_v<T>) {
      /* don't use std::copysign, as it promotes to floating point, which
       * we don't wish to do here */
      z = (y >= U(0)) ? std::abs(x) : -std::abs(x);
    } else {
      z = std::copysign(real(x), real(y));
    }
    return z == x ? g: -g;
  }
};

struct copysign_grad2_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x, const U y)
      const {
    return real(0);
  }
};

struct gamma_p_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real a, const real x) const {
    return Eigen::numext::igamma(a, x);
  }
};

struct gamma_q_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real a, const real x) const {
    return Eigen::numext::igammac(a, x);
  }
};

struct lbeta_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x, const real y) const {
    return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
  }
};

struct lbeta_grad1_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x,
      const real y) const {
    return g*(Eigen::numext::digamma(x) - Eigen::numext::digamma(x + y));
  }
};

struct lbeta_grad2_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x,
      const real y) const {
    return g*(Eigen::numext::digamma(y) - Eigen::numext::digamma(x + y));
  }
};

struct lchoose_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x, const real y) const {
    return std::lgamma(x + real(1)) - std::lgamma(y + real(1)) -
        std::lgamma(x - y + real(1));
  }
};

struct lchoose_grad1_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x,
      const real y) const {
    real d = Eigen::numext::digamma(x - y + real(1));
    real gx = Eigen::numext::digamma(x + real(1)) - d;
    return g*gx;
  }
};

struct lchoose_grad2_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x,
      const real y) const {
    real d = Eigen::numext::digamma(x - y + real(1));
    real gy = -Eigen::numext::digamma(y + real(1)) + d;
    return g*gy;
  }
};

struct pow_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real x, const real y) const {
    return std::pow(x, y);
  }
};

struct pow_grad1_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x,
      const real y) const {
    return g*y*std::pow(x, y - real(1));
  }
};

struct pow_grad2_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const real x,
      const real y) const {
    return g*std::pow(x, y)*std::log(x);
  }
};

struct ibeta_functor {
  NUMBIRCH_HOST_DEVICE real operator()(const real a, const real b,
      const real x) const {
    /* as of Eigen 3.4.0, the edge cases of a == 0 and b == 0 are not handled
    * internally, see https://gitlab.com/libeigen/eigen/-/issues/2359 */
    if (a == real(0) && b != real(0)) {
      return real(1);
    } else if (a != real(0) && b == real(0)) {
      return real(0);
    } else {
      return Eigen::numext::betainc(a, b, x);
    }
  }
};

struct where_functor {
  template<class T, class U, class V>
  NUMBIRCH_HOST_DEVICE promote_t<T,U,V> operator()(const T x, const U y,
      const V z) const {
    using W = promote_t<T,U,V>;
    return x ? W(y) : W(z);
  }
};

struct zero_grad_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x) const {
    return real(0);
  }

  template<class T, class U>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x, const U y)
      const {
    return real(0);
  }
};

template<class T, class>
bool_t<T> logical_not(const T& x) {
  prefetch(x);
  return transform(x, not_functor());
}

template<class T, class>
real_t<T> logical_not_grad(const real_t<T>& g, const bool_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, zero_grad_functor());
}

template<class T, class U, class>
bool_t<T,U> logical_and(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, and_functor());
}

template<class T, class U, class>
real_t<T> logical_and_grad1(const real_t<T,U>& g, const bool_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
real_t<U> logical_and_grad2(const real_t<T,U>& g, const bool_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
bool_t<T,U> logical_or(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, or_functor());
}

template<class T, class U, class>
real_t<T> logical_or_grad1(const real_t<T,U>& g, const bool_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
real_t<U> logical_or_grad2(const real_t<T,U>& g, const bool_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
bool_t<T,U> equal(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, equal_functor());
}

template<class T, class U, class>
real_t<T> equal_grad1(const real_t<T,U>& g, const bool_t<T,U>& z, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
real_t<U> equal_grad2(const real_t<T,U>& g, const bool_t<T,U>& z, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
bool_t<T,U> not_equal(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, not_equal_functor());
}

template<class T, class U, class>
real_t<T> not_equal_grad1(const real_t<T,U>& g, const bool_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
real_t<U> not_equal_grad2(const real_t<T,U>& g, const bool_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
bool_t<T,U> less(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, less_functor());
}

template<class T, class U, class>
real_t<T> less_grad1(const real_t<T,U>& g, const bool_t<T,U>& z, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
real_t<U> less_grad2(const real_t<T,U>& g, const bool_t<T,U>& z, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
bool_t<T,U> less_or_equal(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, less_or_equal_functor());
}

template<class T, class U, class>
real_t<T> less_or_equal_grad1(const real_t<T,U>& g, const bool_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
real_t<U> less_or_equal_grad2(const real_t<T,U>& g, const bool_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
bool_t<T,U> greater(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, greater_functor());
}

template<class T, class U, class>
real_t<T> greater_grad1(const real_t<T,U>& g, const bool_t<T,U>& z, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
real_t<U> greater_grad2(const real_t<T,U>& g, const bool_t<T,U>& z, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
bool_t<T,U> greater_or_equal(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, greater_or_equal_functor());
}

template<class T, class U, class>
real_t<T> greater_or_equal_grad1(const real_t<T,U>& g, const bool_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class U, class>
real_t<U> greater_or_equal_grad2(const real_t<T,U>& g, const bool_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y, zero_grad_functor()));
}

template<class T, class>
T abs(const T& x) {
  prefetch(x);
  return transform(x, abs_functor());
}

template<class T, class>
real_t<T> abs_grad(const real_t<T>& g, const T& y, const T& x) {
  prefetch(x);
  return transform(g, x, abs_grad_functor());
}

template<class T, class>
real_t<T> acos(const T& x) {
  prefetch(x);
  return transform(x, acos_functor());
}

template<class T, class>
real_t<T> acos_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, acos_grad_functor());
}

template<class T, class U, class>
implicit_t<T,U> add(const T& x, const U& y) {
  /* optimizations for addition of scalar zero */
  if constexpr (is_arithmetic_v<T> && std::is_same_v<implicit_t<T,U>,U>) {
    if (x == T(0)) {
      return y;
    }
  } else if constexpr (is_arithmetic_v<U> &&
      std::is_same_v<implicit_t<T,U>,T>) {
    if (y == U(0)) {
      return x;
    }
  }
  prefetch(x);
  prefetch(y);
  return transform(x, y, add_functor());
}

template<class T, class U, class>
real_t<T> add_grad1(const real_t<T,U>& g, const implicit_t<T,U>& z, const T& x,
    const U& y) {
  return aggregate<dimension_v<T>>(g);
}

template<class T, class U, class>
real_t<U> add_grad2(const real_t<T,U>& g, const implicit_t<T,U>& z, const T& x,
    const U& y) {
  return aggregate<dimension_v<U>>(g);
}

template<class T, class>
real_t<T> asin(const T& x) {
  prefetch(x);
  return transform(x, asin_functor());
}

template<class T, class>
real_t<T> asin_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, asin_grad_functor());
}

template<class T, class>
real_t<T> atan(const T& x) {
  prefetch(x);
  return transform(x, atan_functor());
}

template<class T, class>
real_t<T> atan_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, atan_grad_functor());
}

template<class R, class T, class>
explicit_t<R,T> cast(const T& x) {
  prefetch(x);
  return transform(x, cast_functor<R>());
}

template<class T, class>
T ceil(const T& x) {
  prefetch(x);
  return transform(x, ceil_functor());
}

template<class T, class>
real_t<T> ceil_grad(const real_t<T>& g, const T& y, const T& x) {
  prefetch(x);
  return transform(g, x, zero_grad_functor());
}

template<class T, class U, class>
implicit_t<T,U> copysign(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, copysign_functor());
}

template<class T, class U, class>
real_t<T> copysign_grad1(const real_t<T,U>& g, const implicit_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y,
      copysign_grad1_functor()));
}

template<class T, class U, class>
real_t<U> copysign_grad2(const real_t<T,U>& g, const implicit_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y,
      copysign_grad2_functor()));
}

template<class T, class>
real_t<T> cos(const T& x) {
  prefetch(x);
  return transform(x, cos_functor());
}

template<class T, class>
real_t<T> cos_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, cos_grad_functor());
}

template<class T, class>
real_t<T> cosh(const T& x) {
  prefetch(x);
  return transform(x, cosh_functor());
}

template<class T, class>
real_t<T> cosh_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, cosh_grad_functor());
}

template<class T, class>
real_t<T> digamma(const T& x) {
  prefetch(x);
  return transform(x, digamma_functor());
}

template<class T, class U, class>
real_t<T,U> digamma(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, digamma_functor());
}

template<class T, class U, class>
implicit_t<T,U> div(const T& x, const U& y) {
  /* optimization for division of scalar one */
  if constexpr (is_arithmetic_v<U> && std::is_same_v<implicit_t<T,U>,T>) {
    if (y == U(1)) {
      return x;
    }
  }

  prefetch(x);
  prefetch(y);
  return transform(x, y, div_functor());
}

template<class T, class U, class>
real_t<T> div_grad1(const real_t<T,U>& g, const implicit_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y, div_grad1_functor()));
}

template<class T, class U, class>
real_t<U> div_grad2(const real_t<T,U>& g, const implicit_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y, div_grad2_functor()));
}

template<class T, class>
real_t<T> exp(const T& x) {
  prefetch(x);
  return transform(x, exp_functor());
}

template<class T, class>
real_t<T> exp_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  return hadamard(g, y);
}

template<class T, class>
real_t<T> expm1(const T& x) {
  prefetch(x);
  return transform(x, expm1_functor());
}

template<class T, class>
real_t<T> expm1_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  return hadamard(g, y);
}

template<class T, class>
T floor(const T& x) {
  prefetch(x);
  return transform(x, floor_functor());
}

template<class T, class>
real_t<T> floor_grad(const real_t<T>& g, const T& y, const T& x) {
  prefetch(x);
  return transform(g, x, zero_grad_functor());
}

template<class T, class U, class>
real_t<T,U> gamma_p(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, gamma_p_functor());
}

template<class T, class U, class>
real_t<T,U> gamma_q(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, gamma_q_functor());
}

template<class T, class U, class V, class>
real_t<T,U,V> ibeta(const T& x, const U& y, const V& z) {
  prefetch(x);
  prefetch(y);
  prefetch(z);
  return transform(x, y, z, ibeta_functor());
}

template<class T, class>
bool_t<T> isfinite(const T& x) {
  prefetch(x);
  return transform(x, isfinite_functor());
}

template<class T, class>
real_t<T> isfinite_grad(const real_t<T>& g, const bool_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, zero_grad_functor());
}

template<class T, class>
bool_t<T> isinf(const T& x) {
  prefetch(x);
  return transform(x, isinf_functor());
}

template<class T, class>
real_t<T> isinf_grad(const real_t<T>& g, const bool_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, zero_grad_functor());
}

template<class T, class>
bool_t<T> isnan(const T& x) {
  prefetch(x);
  return transform(x, isnan_functor());
}

template<class T, class>
real_t<T> isnan_grad(const real_t<T>& g, const bool_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, zero_grad_functor());
}

template<class T, class U, class>
real_t<T,U> lbeta(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, lbeta_functor());
}

template<class T, class U, class>
real_t<T> lbeta_grad1(const real_t<T,U>& g, const real_t<T,U>& z, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y, lbeta_grad1_functor()));
}

template<class T, class U, class>
real_t<U> lbeta_grad2(const real_t<T,U>& g, const real_t<T,U>& z, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y, lbeta_grad2_functor()));
}

template<class T, class U, class>
real_t<T,U> lchoose(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, lchoose_functor());
}

template<class T, class U, class>
real_t<T> lchoose_grad1(const real_t<T,U>& g, const real_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y,
      lchoose_grad1_functor()));
}

template<class T, class U, class>
real_t<U> lchoose_grad2(const real_t<T,U>& g, const real_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y,
      lchoose_grad2_functor()));
}

template<class T, class>
real_t<T> lfact(const T& x) {
  prefetch(x);
  return transform(x, lfact_functor());
}

template<class T, class>
real_t<T> lfact_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, lfact_grad_functor());
}

template<class T, class>
real_t<T> lgamma(const T& x) {
  prefetch(x);
  return transform(x, lgamma_functor());
}

template<class T, class>
real_t<T> lgamma_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x,
      lgamma_grad_functor());
}

template<class T, class U, class>
real_t<T,U> lgamma(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, lgamma_functor());
}

template<class T, class U, class>
real_t<T> lgamma_grad1(const real_t<T,U>& g, const real_t<T,U>& z, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y,
      lgamma_grad1_functor()));
}

template<class T, class U, class>
real_t<U> lgamma_grad2(const real_t<T,U>& g, const real_t<T,U>& z, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y,
      lgamma_grad2_functor()));
}

template<class T, class>
real_t<T> log(const T& x) {
  prefetch(x);
  return transform(x, log_functor());
}

template<class T, class>
real_t<T> log_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, log_grad_functor());
}

template<class T, class>
real_t<T> log1p(const T& x) {
  prefetch(x);
  return transform(x, log1p_functor());
}

template<class T, class>
real_t<T> log1p_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, log1p_grad_functor());
}

template<class T, class U, class>
implicit_t<T,U> hadamard(const T& x, const U& y) {
  /* optimizations for multiplication of scalar one */
  if constexpr (is_arithmetic_v<T> && std::is_same_v<implicit_t<T,U>,U>) {
    if (x == T(1)) {
      return y;
    }
  } else if constexpr (is_arithmetic_v<U> &&
      std::is_same_v<implicit_t<T,U>,T>) {
    if (y == U(1)) {
      return x;
    }
  }

  prefetch(x);
  prefetch(y);
  return transform(x, y, hadamard_functor());
}

template<class T, class U, class>
real_t<T> hadamard_grad1(const real_t<T,U>& g, const implicit_t<T,U>& z,
    const T& x, const U& y) {
  /* optimization for multiplication of scalar one */
  if constexpr (is_arithmetic_v<U>) {
    if (y == U(1)) {
      return g;
    }
  }

  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y,
      hadamard_grad1_functor()));
}

template<class T, class U, class>
real_t<U> hadamard_grad2(const real_t<T,U>& g, const implicit_t<T,U>& z,
    const T& x, const U& y) {
  /* optimization for multiplication of scalar one */
  if constexpr (is_arithmetic_v<T>) {
    if (x == T(1)) {
      return g;
    }
  }

  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y,
      hadamard_grad2_functor()));
}

template<class T, class>
T neg(const T& x) {
  prefetch(x);
  return transform(x, neg_functor());
}

template<class T, class>
real_t<T> neg_grad(const real_t<T>& g, const T& y, const T& x) {
  return neg(g);
}

template<class T, class U, class>
real_t<T,U> pow(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, pow_functor());
}

template<class T, class U, class>
real_t<T> pow_grad1(const real_t<T,U>& g, const real_t<T,U>& z, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(transform(g, x, y, pow_grad1_functor()));
}

template<class T, class U, class>
real_t<U> pow_grad2(const real_t<T,U>& g, const real_t<T,U>& z, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<U>>(transform(g, x, y, pow_grad2_functor()));
}

template<class T, class>
T rectify(const T& x) {
  prefetch(x);
  return transform(x, rectify_functor());
}

template<class T, class>
real_t<T> rectify_grad(const real_t<T>& g, const T& y, const T& x) {
  prefetch(x);
  return transform(g, x, rectify_grad_functor());
}

template<class T, class>
T round(const T& x) {
  prefetch(x);
  return transform(x, round_functor());
}

template<class T, class>
real_t<T> round_grad(const real_t<T>& g, const T& y, const T& x) {
  prefetch(x);
  return transform(g, x, zero_grad_functor());
}

template<class T, class>
real_t<T> sin(const T& x) {
  prefetch(x);
  return transform(x, sin_functor());
}

template<class T, class>
real_t<T> sin_grad(const real_t<T>& g, const real_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, sin_grad_functor());
}

template<class T, class>
real_t<T> sinh(const T& x) {
  prefetch(x);
  return transform(x, sinh_functor());
}

template<class T, class>
real_t<T> sinh_grad(const real_t<T>& g, const real_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, sinh_grad_functor());
}

template<class T, class>
real_t<T> sqrt(const T& x) {
  prefetch(x);
  return transform(x, sqrt_functor());
}

template<class T, class>
real_t<T> sqrt_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, sqrt_grad_functor());
}

template<class T, class U, class>
implicit_t<T,U> sub(const T& x, const U& y) {
  /* optimization for subtraction of scalar zero */
  if constexpr (is_arithmetic_v<U> && std::is_same_v<implicit_t<T,U>,T>) {
    if (y == U(0)) {
      return x;
    }
  }

  prefetch(x);
  prefetch(y);
  return transform(x, y, sub_functor());
}

template<class T, class U, class>
real_t<T> sub_grad1(const real_t<T,U>& g, const implicit_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return aggregate<dimension_v<T>>(g);
}

template<class T, class U, class>
real_t<U> sub_grad2(const real_t<T,U>& g, const implicit_t<T,U>& z,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return neg(aggregate<dimension_v<U>>(g));
}

template<class T, class>
real_t<T> tan(const T& x) {
  prefetch(x);
  return transform(x, tan_functor());
}

template<class T, class>
real_t<T> tan_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, tan_grad_functor());
}

template<class T, class>
real_t<T> tanh(const T& x) {
  prefetch(x);
  return transform(x, tanh_functor());
}

template<class T, class>
real_t<T> tanh_grad(const real_t<T>& g, const real_t<T>& y, const T& x) {
  prefetch(x);
  return transform(g, x, tanh_grad_functor());
}

template<class T, class U, class V, class>
implicit_t<T,U,V> where(const T& x, const U& y, const V& z) {
  prefetch(x);
  prefetch(y);
  prefetch(z);
  return transform(x, y, z, where_functor());
}

template<class T, class U, class V, class>
real_t<T> where_grad1(const real_t<U,V>& g, const implicit_t<T,U,V>& r,
    const T& x, const U& y, const V& z) {
  return Array(shape(x), real(0));
}

template<class T, class U, class V, class>
real_t<U> where_grad2(const real_t<U,V>& g, const implicit_t<T,U,V>& r,
    const T& x, const U& y, const V& z) {
  return aggregate<dimension_v<U>>(where(x, g, real(0)));
}

template<class T, class U, class V, class>
real_t<V> where_grad3(const real_t<U,V>& g, const implicit_t<T,U,V>& r,
    const T& x, const U& y, const V& z) {
  return aggregate<dimension_v<V>>(where(x, real(0), g));
}

}
