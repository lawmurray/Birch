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
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return x;
  }
};

template<class R>
struct negate_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return -x;
  }
};

template<class R>
struct add_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x, const R y) const {
    return x + y;
  }
};

template<class R>
struct subtract_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x, const R y) const {
    return x - y;
  }
};

template<class R, class T>
struct scalar_divide_functor {
  T y;
  scalar_divide_functor(const T& y) : y(y) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    auto z = R(element(y));
    return x/z;
  }
};

template<class R, class T>
struct scalar_multiply_functor {
  T y;
  scalar_multiply_functor(const T& y) : y(y) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    auto z = R(element(y));
    return x*z;
  }
};

template<class R>
struct not_functor {
  template<class T>
  NUMBIRCH_HOST_DEVICE R operator()(const T x) const {
    return !x;
  }
};

template<class R>
struct not_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return R(0);
  }
};

template<class R>
struct and_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T x, const U y) const {
    return x && y;
  }
};

template<class R>
struct and_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    return pair<R,R>{R(0), R(0)};
  }
};

template<class R>
struct or_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T x, const U y) const {
    return x || y;
  }
};

template<class R>
struct or_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    return pair<R,R>{R(0), R(0)};
  }
};

template<class R>
struct equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T x, const U y) const {
    return x == y;
  }
};

template<class R>
struct equal_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    return pair<R,R>{R(0), R(0)};
  }
};


template<class R>
struct not_equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T x, const U y) const {
    return x != y;
  }
};

template<class R>
struct not_equal_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    return pair<R,R>{R(0), R(0)};
  }
};


template<class R>
struct less_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T x, const U y) const {
    return x < y;
  }
};

template<class R>
struct less_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    return pair<R,R>{R(0), R(0)};
  }
};


template<class R>
struct less_or_equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T x, const U y) const {
    return x <= y;
  }
};

template<class R>
struct less_or_equal_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    return pair<R,R>{R(0), R(0)};
  }
};

template<class R>
struct greater_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T x, const U y) const {
    return x > y;
  }
};

template<class R>
struct greater_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    return pair<R,R>{R(0), R(0)};
  }
};


template<class R>
struct greater_or_equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T x, const U y) const {
    return x >= y;
  }
};

template<class R>
struct greater_or_equal_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    return pair<R,R>{R(0), R(0)};
  }
};


template<class R>
struct abs_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::abs(x);
  }
};

template<class R>
struct abs_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    if constexpr (is_integral_v<R>) {
      // don't use std::copysign, as it promotes to floating point, which
      // we don't wish to do here
      return (x >= R(0)) ? std::abs(g) : -std::abs(g);
    } else {
      return std::copysign(g, x);
    }
  }
};

template<class R>
struct acos_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::acos(x);
  }
};

template<class R>
struct acos_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return -g/std::sqrt(R(1.0) - x*x);
  }
};

template<class R>
struct asin_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::asin(x);
  }
};

template<class R>
struct asin_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g/std::sqrt(R(1.0) - x*x);
  }
};

template<class R>
struct atan_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::atan(x);
  }
};

template<class R>
struct atan_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g/(R(1.0) + x*x);
  }
};

template<class R>
struct ceil_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    if constexpr (is_integral_v<R>) {
      return x;
    } else {
      return std::ceil(x);
    }
  }
};

template<class R>
struct ceil_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return R(0.0);
  }
};

template<class R>
struct cos_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::cos(x);
  }
};

template<class R>
struct cos_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return -g*std::sin(x);
  }
};

template<class R>
struct cosh_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::cosh(x);
  }
};

template<class R>
struct cosh_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return -g*std::sinh(x);
  }
};

template<class R>
struct count_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return (x == 0) ? 0 : 1;
  }
};

template<class R>
struct count_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return R(0);
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

template<class R>
struct digamma_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return Eigen::numext::digamma(x);
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R x, const R y) const {
    R z = 0;
    for (R i = 1; i <= y; ++i) {
      z += Eigen::numext::digamma(x + R(0.5)*(1 - i));
    }
    return z;
  }
};

template<class R>
struct exp_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::exp(x);
  }
};

template<class R>
struct exp_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g*std::exp(x);
  }
};

template<class R>
struct expm1_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::expm1(x);
  }
};

template<class R>
struct expm1_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g*std::expm1(x);
  }
};

template<class R>
struct floor_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    if constexpr (is_integral_v<R>) {
      return x;
    } else {
      return std::floor(x);
    }
  }
};

template<class R>
struct floor_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return R(0.0);
  }
};

template<class R>
struct hadamard_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x, const R y) const {
    return x*y;
  }
};

template<class R>
struct hadamard_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    return pair<R,R>{g*y, g*x};
  }
};

template<class R>
struct lfact_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::lgamma(x + R(1));
  }
};

template<class R>
struct lfact_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g*Eigen::numext::digamma(x + R(1));
  }
};

template<class R>
struct lgamma_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::lgamma(x);
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R x, const R y) const {
    R z = R(0.25)*y*(y - R(1))*std::log(R(PI));
    for (R i = R(1); i <= y; ++i) {
      z += std::lgamma(x + R(0.5)*(R(1) - i));
    }
    return z;
  }
};

template<class R>
struct lgamma_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g*Eigen::numext::digamma(x);
  }
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    R z = 0;
    for (R i = 1; i <= y; ++i) {
      z += Eigen::numext::digamma(x + R(0.5)*(1 - i));
    }
    return pair<R,R>{g*z, R(0)};
  }
};

template<class R>
struct log_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::log(x);
  }
};

template<class R>
struct log_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g/x;
  }
};

template<class R>
struct log1p_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::log1p(x);
  }
};

template<class R>
struct log1p_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g/(x + R(1.0));
  }
};

template<class R>
struct log_abs_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::log(std::abs(x));
  }
};

template<class R>
struct log_square_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return R(2)*std::log(x);
  }
};

template<class R>
struct rectify_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::max(R(0), x);
  }
};

template<class R>
struct rectify_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return (x > R(0)) ? g : R(0);
  }
};

template<class R>
struct round_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    if constexpr (is_integral_v<R>) {
      return x;
    } else {
      return std::round(x);
    }
  }
};

template<class R>
struct round_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return R(0.0);
  }
};

template<class R>
struct sin_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::sin(x);
  }
};

template<class R>
struct sin_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g*std::cos(x);
  }
};

template<class R>
struct sinh_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::sinh(x);
  }
};

template<class R>
struct sinh_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g*std::cosh(x);
  }
};

template<class R>
struct sqrt_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::sqrt(x);
  }
};

template<class R>
struct sqrt_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g*R(0.5)/std::sqrt(x);
  }
};

template<class R>
struct tan_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::tan(x);
  }
};

template<class R>
struct tan_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g*(R(1.0) + std::pow(std::tan(x), R(2.0)));
  }
};

template<class R>
struct tanh_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::tanh(x);
  }
};

template<class R>
struct tanh_grad_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R g, const R x) const {
    return g*(R(1.0) + std::pow(std::tanh(x), R(2.0)));
  }
};

template<class R>
struct copysign_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x, const R y) const {
    if constexpr (is_integral_v<R>) {
      // don't use std::copysign, as it promotes to floating point, which
      // we don't wish to do here
      return (y >= R(0)) ? std::abs(x) : -std::abs(x);
    } else {
      return std::copysign(x, y);
    }
  }
};

template<class R>
struct copysign_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    return pair<R,R>{copysign(x, y) == x ? g: -g, R(0)};
  }
};

template<class R>
struct gamma_p_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R a, const R x) const {
    return Eigen::numext::igamma(a, x);
  }
};

template<class R>
struct gamma_q_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R a, const R x) const {
    return Eigen::numext::igammac(a, x);
  }
};

template<class R>
struct lbeta_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x, const R y) const {
    return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
  }
};

template<class R>
struct lbeta_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    R d = Eigen::numext::digamma(x + y);
    R gx = Eigen::numext::digamma(x) - d;
    R gy = Eigen::numext::digamma(y) - d;
    return pair<R,R>{g*gx, g*gy};
  }
};

template<class R>
struct lchoose_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x, const R y) const {
    return std::lgamma(x + R(1)) - std::lgamma(y + R(1)) -
        std::lgamma(x - y + R(1));
  }
};

template<class R>
struct lchoose_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    R d = Eigen::numext::digamma(x - y + R(1));
    R gx = Eigen::numext::digamma(x + R(1)) - d;
    R gy = -Eigen::numext::digamma(y + R(1)) + d;
    return pair<R,R>{g*gx, g*gy};
  }
};

template<class R>
struct pow_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x, const R y) const {
    return std::pow(x, y);
  }
};

template<class R>
struct pow_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<R,R> operator()(const R g, const R x, const R y)
      const {
    R gx = y*std::pow(x, y - R(1));
    R gy = std::pow(x, y)*std::log(x);
    return pair<R,R>{g*gx, g*gy};
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

template<class R, class T>
struct sum_grad_functor {
  T g;
  sum_grad_functor(const T& g) : g(g) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return R(element(g));
  }
};

template<class R>
struct ibeta_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R a, const R b, const R x) const {
    /* as of Eigen 3.4.0, the edge cases of a == 0 and b == 0 are not handled
    * internally, see https://gitlab.com/libeigen/eigen/-/issues/2359 */
    if (a == R(0) && b != R(0)) {
      return R(1);
    } else if (a != R(0) && b == R(0)) {
      return R(0);
    } else {
      return Eigen::numext::betainc(a, b, x);
    }
  }
};

}
