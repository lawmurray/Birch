/**
 * @file
 */
#pragma once

#include "numbirch/common/element.hpp"

#if defined(HAVE_UNSUPPORTED_EIGEN_SPECIALFUNCTIONS)
#include <unsupported/Eigen/SpecialFunctions>
#elif defined(HAVE_EIGEN3_UNSUPPORTED_EIGEN_SPECIALFUNCTIONS)
#include <eigen3/unsupported/Eigen/SpecialFunctions>
#endif

namespace numbirch {
static const long double PI = 3.1415926535897932384626433832795;

template<class R>
struct cast_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const T x) const {
    return R(x);
  }
};

struct negate_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const T x) const {
    return -x;
  }
};

struct add_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE auto operator()(const T x, const U y) const {
    using R = promote_t<T,U>;
    return R(x) + R(y);
  }
};

struct subtract_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE auto operator()(const T x, const U y) const {
    using R = promote_t<T,U>;
    return R(x) - R(y);
  }
};

template<class U>
struct scalar_divide_functor {
  U y;
  scalar_divide_functor(const U& y) : y(y) {
    //
  }
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const T x) const {
    using R = promote_t<T,std::decay_t<decltype(element(y))>>;
    return R(x)/R(element(y));
  }
};

template<class U>
struct scalar_multiply_functor {
  U y;
  scalar_multiply_functor(const U& y) : y(y) {
    //
  }
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const T x) const {
    using R = promote_t<T,std::decay_t<decltype(element(y))>>;
    return R(x)*R(element(y));
  }
};

struct not_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x) const {
    return bool(!x);
  }
};

struct not_grad_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x) const {
    return real(0);
  }
};

struct and_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x && y);
  }
};

struct and_grad_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const T x,
      const U y) const {
    return pair<real,real>{real(0), real(0)};
  }
};

struct or_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x || y);
  }
};

struct or_grad_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const T x,
      const U y) const {
    return pair<real,real>{real(0), real(0)};
  }
};

struct equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x == y);
  }
};

struct equal_grad_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const T x,
      const U y) const {
    return pair<real,real>{real(0), real(0)};
  }
};


struct not_equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x != y);
  }
};

struct not_equal_grad_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const T x,
      const U y) const {
    return pair<real,real>{real(0), real(0)};
  }
};

struct less_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x < y);
  }
};

struct less_grad_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const T x,
      const U y) const {
    return pair<real,real>{real(0), real(0)};
  }
};

struct less_or_equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x <= y);
  }
};

struct less_or_equal_grad_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const T x,
      const U y) const {
    return pair<real,real>{real(0), real(0)};
  }
};

struct greater_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x > y);
  }
};

struct greater_grad_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const T x,
      const U y) const {
    return pair<real,real>{real(0), real(0)};
  }
};

struct greater_or_equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const U y) const {
    return bool(x >= y);
  }
};

struct greater_or_equal_grad_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const T x,
      const U y) const {
    return pair<real,real>{real(0), real(0)};
  }
};

struct abs_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const T x) const {
    return std::abs(x);
  }
};

struct abs_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return std::copysign(g, x);
  }
};

struct acos_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::acos(x);
  }
};

struct acos_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return -g/std::sqrt(real(1.0) - x*x);
  }
};

struct asin_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::asin(x);
  }
};

struct asin_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return g/std::sqrt(real(1) - x*x);
  }
};

struct atan_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::atan(x);
  }
};

struct atan_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return g/(real(1) + x*x);
  }
};

struct ceil_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const T x) const {
    if constexpr (is_integral_v<T>) {
      return x;
    } else {
      return std::ceil(x);
    }
  }
};

struct ceil_grad_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const T x) const {
    return real(0.0);
  }
};

struct cos_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::cos(x);
  }
};

struct cos_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return -g*std::sin(x);
  }
};

struct cosh_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::cosh(x);
  }
};

struct cosh_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return -g*std::sinh(x);
  }
};

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

template<class T>
struct diagonal_functor {
  const T a;
  diagonal_functor(const T a) :
      a(a) {
    //
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const int i, const int j) const {
    return (i == j) ? element(a) : 0;
  }
};

struct digamma_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return Eigen::numext::digamma(x);
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const real x, const int y) const {
    real z = 0;
    for (int i = 1; i <= y; ++i) {
      z += Eigen::numext::digamma(x + real(0.5)*(1 - i));
    }
    return z;
  }
};

struct exp_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::exp(x);
  }
};

struct expm1_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::expm1(x);
  }
};

struct floor_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const T x) const {
    if constexpr (is_integral_v<T>) {
      return x;
    } else {
      return std::floor(x);
    }
  }
};

struct floor_grad_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const T x) const {
    return real(0);
  }
};

struct hadamard_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE auto operator()(const T x, const U y) const {
    return x*y;
  }
};

struct hadamard_grad_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const T x,
      const U y) const {
    return pair<real,real>{g*y, g*x};
  }
};

struct lfact_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::lgamma(x + real(1));
  }
};

struct lfact_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return g*Eigen::numext::digamma(x + real(1));
  }
};

struct lgamma_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::lgamma(x);
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const real x, const real y) const {
    real z = real(0.25)*y*(y - 1)*std::log(real(PI));
    for (int i = 1; i <= y; ++i) {
      z += std::lgamma(x + real(0.5)*(1 - i));
    }
    return z;
  }
};

struct lgamma_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return g*Eigen::numext::digamma(x);
  }
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const real x,
      const int y) const {
    real z = 0.0;
    for (int i = 1; i <= y; ++i) {
      z += Eigen::numext::digamma(x + real(0.5)*(1 - i));
    }
    return pair<real,real>{g*z, real(0)};
  }
};

struct log_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::log(x);
  }
};

struct log_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return g/x;
  }
};

struct log1p_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::log1p(x);
  }
};

struct log1p_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return g/(x + real(1.0));
  }
};

struct log_abs_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::log(std::abs(x));
  }
};

struct log_square_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return real(2)*std::log(x);
  }
};

struct rectify_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const T x) const {
    return std::max(T(0), x);
  }
};

struct rectify_grad_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE real operator()(const real g, const T x) const {
    return (x > T(0)) ? g : real(0);
  }
};

struct round_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const T x) const {
    if constexpr (is_integral_v<T>) {
      return x;
    } else {
      return std::round(x);
    }
  }
};

struct round_grad_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const T x) const {
    return real(0);
  }
};

struct sin_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::sin(x);
  }
};

struct sin_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return g*std::cos(x);
  }
};

struct sinh_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::sinh(x);
  }
};

struct sinh_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return g*std::cosh(x);
  }
};

struct sqrt_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::sqrt(x);
  }
};

struct sqrt_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return g*real(0.5)/std::sqrt(x);
  }
};

struct tan_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::tan(x);
  }
};

struct tan_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return g*(real(1) + std::pow(std::tan(x), real(2)));
  }
};

struct tanh_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x) const {
    return std::tanh(x);
  }
};

struct tanh_grad_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real g, const real x) const {
    return g*(real(1) + std::pow(std::tanh(x), real(2)));
  }
};

struct copysign_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const U y) const {
    if constexpr (is_integral_v<T>) {
      // don't use std::copysign, as it promotes to floating point, which
      // we don't wish to do here
      return (y >= U(0)) ? std::abs(x) : -std::abs(x);
    } else {
      return std::copysign(real(x), real(y));
    }
  }
};

struct copysign_grad_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const T x,
      const U y) const {
    T z;
    if constexpr (is_integral_v<T>) {
      // don't use std::copysign, as it promotes to floating point, which
      // we don't wish to do here
      z = (y >= U(0)) ? std::abs(x) : -std::abs(x);
    } else {
      z = std::copysign(real(x), real(y));
    }
    return pair<real,real>{z == x ? g: -g, real(0)};
  }
};

struct gamma_p_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real a, const real x) const {
    return Eigen::numext::igamma(a, x);
  }
};

struct gamma_q_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real a, const real x) const {
    return Eigen::numext::igammac(a, x);
  }
};

struct lbeta_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x, const real y) const {
    return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
  }
};

struct lbeta_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const real x,
      const real y) const {
    real d = Eigen::numext::digamma(x + y);
    real gx = Eigen::numext::digamma(x) - d;
    real gy = Eigen::numext::digamma(y) - d;
    return pair<real,real>{g*gx, g*gy};
  }
};

struct lchoose_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x, const real y) const {
    return std::lgamma(x + real(1)) - std::lgamma(y + real(1)) -
        std::lgamma(x - y + real(1));
  }
};

struct lchoose_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const real x,
      const real y) const {
    real d = Eigen::numext::digamma(x - y + real(1));
    real gx = Eigen::numext::digamma(x + real(1)) - d;
    real gy = -Eigen::numext::digamma(y + real(1)) + d;
    return pair<real,real>{g*gx, g*gy};
  }
};

struct pow_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real x, const real y) const {
    return std::pow(x, y);
  }
};

struct pow_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<real,real> operator()(const real g, const real x,
      const real y) const {
    real gx = y*std::pow(x, y - real(1));
    real gy = std::pow(x, y)*std::log(x);
    return pair<real,real>{g*gx, g*gy};
  }
};

template<class T, class U, class V = int>
struct single_functor {
  const T x;
  const U k;
  const V l;
  single_functor(const T& x, const U& k, const V& l = 1) :
      x(x), k(k), l(l) {
    //
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const int i, const int j) const {
    return (i == element(k) - 1 && j == element(l) - 1) ? element(x) : 0;
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
    return element(g);
  }
};

struct ibeta_functor {
  NUMBIRCH_HOST_DEVICE auto operator()(const real a, const real b,
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

}
