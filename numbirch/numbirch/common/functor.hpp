/**
 * @file
 */
#pragma once

#include "numbirch/type.hpp"

/* for recent versions of CUDA, disables warnings about diag_suppress being
 * deprecated in favor of nv_diag_suppress */
#pragma nv_diag_suppress 20236

#if defined(HAVE_UNSUPPORTED_EIGEN_SPECIALFUNCTIONS)
#include <unsupported/Eigen/SpecialFunctions>
#elif defined(HAVE_EIGEN3_UNSUPPORTED_EIGEN_SPECIALFUNCTIONS)
#include <eigen3/unsupported/Eigen/SpecialFunctions>
#endif

namespace numbirch {
static const long double PI = 3.1415926535897932384626433832795;

template<class R>
struct identity_functor {
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

template<class R>
struct multiply_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x, const R y) const {
    return x*y;
  }
};

template<class R>
struct divide_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x, const R y) const {
    return x/y;
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
struct and_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T x, const U y) const {
    return x && y;
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
struct equal_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T x, const U y) const {
    return x == y;
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
struct less_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T x, const U y) const {
    return x < y;
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
struct greater_functor {
  template<class T, class U>
  NUMBIRCH_HOST_DEVICE R operator()(const T x, const U y) const {
    return x > y;
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
struct abs_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::abs(x);
  }
};

template<class R>
struct acos_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::acos(x);
  }
};

template<class R>
struct asin_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::asin(x);
  }
};

template<class R>
struct atan_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::atan(x);
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
struct cos_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::cos(x);
  }
};

template<class R>
struct cosh_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::cosh(x);
  }
};

template<class R>
struct count_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return (x == 0) ? 0 : 1;
  }
};

template<class R, class U>
struct diagonal_functor {
  diagonal_functor(const U a) :
      a(a) {
    //
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const int i, const int j) const {
    return (i == j) ? R(element(a)) : 0;
  }
  const U a;
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
struct expm1_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::expm1(x);
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
struct log_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::log(x);
  }
};

template<class R>
struct log1p_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::log1p(x);
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
struct rcp_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return R(1)/x;
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
struct sin_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::sin(x);
  }
};

template<class R>
struct sinh_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::sinh(x);
  }
};

template<class R>
struct sqrt_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::sqrt(x);
  }
};

template<class R>
struct tan_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::tan(x);
  }
};

template<class R>
struct tanh_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x) const {
    return std::tanh(x);
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
    R gx = Eigen::numext::digamma(x + R(1)) -
        Eigen::numext::digamma(x - y + R(1));
    R gy = -Eigen::numext::digamma(y + R(1)) +
        Eigen::numext::digamma(x - y + R(1));
    return pair<R,R>{g*gx, g*gy};
  }
};

template<class R>
struct pow_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const R x, const R y) const {
    return std::pow(x, y);
  }
};

template<class R, class U, class V = int>
struct single_functor {
  single_functor(const U k, const V l = 1) :
      k(k), l(l) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const int i, const int j) const {
    return (i == element(k) - 1 && j == element(l) - 1) ? R(1) : R(0);
  }
  const U k;
  const V l;
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
