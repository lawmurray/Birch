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

template<class T>
struct identity_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return x;
  }
};

template<class T>
struct negate_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return -x;
  }
};

template<class T>
struct add_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<class T>
struct subtract_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return x - y;
  }
};

template<class T>
struct multiply_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return x*y;
  }
};

template<class T>
struct divide_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return x/y;
  }
};

template<class T>
struct not_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return !x;
  }
};

template<class T>
struct and_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return x && y;
  }
};

template<class T>
struct or_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return x || y;
  }
};

template<class T>
struct equal_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return x == y;
  }
};

template<class T>
struct not_equal_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return x != y;
  }
};

template<class T>
struct less_functor {
  NUMBIRCH_HOST_DEVICE bool operator()(const T x, const T y) const {
    return x < y;
  }
};

template<class T>
struct less_or_equal_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return x <= y;
  }
};

template<class T>
struct greater_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return x > y;
  }
};

template<class T>
struct greater_or_equal_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return x >= y;
  }
};

template<class T>
struct abs_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::abs(x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct acos_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::acos(x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct asin_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::asin(x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct atan_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::atan(x);
  }
};

template<class T>
struct ceil_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    if constexpr (is_integral_v<T>) {
      return x;
    } else {
      return std::ceil(x);
    }
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct cos_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::cos(x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct cosh_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::cosh(x);
  }
};

template<class T>
struct count_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return (x == 0) ? 0 : 1;
  }
};

template<class T, class U>
struct diagonal_functor {
  diagonal_functor(const U a) :
      a(a) {
    //
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const int i, const int j) const {
    return (i == j) ? T(element(a)) : 0;
  }
  const U a;
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct digamma_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return Eigen::numext::digamma(x);
  }
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    T z = 0;
    for (T i = 1; i <= y; ++i) {
      z += Eigen::numext::digamma(x + T(0.5)*(1 - i));
    }
    return z;
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct exp_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::exp(x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct expm1_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::expm1(x);
  }
};

template<class T>
struct floor_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    if constexpr (is_integral_v<T>) {
      return x;
    } else {
      return std::floor(x);
    }
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct lfact_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::lgamma(x + T(1));
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct lfact_grad_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T g, const T x) const {
    return g*Eigen::numext::digamma(x + T(1));
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct lgamma_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::lgamma(x);
  }
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    T z = T(0.25)*y*(y - T(1))*std::log(T(PI));
    for (T i = T(1); i <= y; ++i) {
      z += std::lgamma(x + T(0.5)*(T(1) - i));
    }
    return z;
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct log_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::log(x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct log1p_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::log1p(x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct log_abs_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::log(std::abs(x));
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct log_square_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return T(2)*std::log(x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct rcp_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return T(1)/x;
  }
};

template<class T>
struct rectify_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::max(T(0), x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct rectify_grad_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T g, const T x) const {
    return (x > T(0)) ? g : T(0);
  }
};

template<class T>
struct round_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    if constexpr (is_integral_v<T>) {
      return x;
    } else {
      return std::round(x);
    }
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct sin_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::sin(x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct sinh_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::sinh(x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct sqrt_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::sqrt(x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct tan_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::tan(x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct tanh_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x) const {
    return std::tanh(x);
  }
};

template<class T>
struct copysign_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    if constexpr (is_integral_v<T>) {
      // don't use std::copysign, as it promotes to floating point, which
      // we don't wish to do here
      return (y >= T(0)) ? std::abs(x) : -std::abs(x);
    } else {
      return std::copysign(x, y);
    }
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct gamma_p_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T a, const T x) const {
    return Eigen::numext::igamma(a, x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct gamma_q_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T a, const T x) const {
    return Eigen::numext::igammac(a, x);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct lbeta_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct lchoose_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const int x, const int y) const {
    return std::lgamma(x + T(1)) - std::lgamma(y + T(1)) -
        std::lgamma(x - y + T(1));
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct lchoose_grad_functor {
  NUMBIRCH_HOST_DEVICE pair<T,T> operator()(const T g, const T x, const T y)
      const {
    T gx = Eigen::numext::digamma(x + T(1)) -
        Eigen::numext::digamma(x - y + T(1));
    T gy = -Eigen::numext::digamma(y + T(1)) +
        Eigen::numext::digamma(x - y + T(1));
    return pair<T,T>{g*gx, g*gy};
  }
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct pow_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T x, const T y) const {
    return std::pow(x, y);
  }
};

template<class T, class U, class V = int>
struct single_functor {
  single_functor(const U k, const V l = 1) :
      k(k), l(l) {
    //
  }
  NUMBIRCH_HOST_DEVICE T operator()(const int i, const int j) const {
    return (i == element(k) - 1 && j == element(l) - 1) ? T(1) : T(0);
  }
  const U k;
  const V l;
};

template<class T, class = std::enable_if_t<is_floating_point_v<T>,int>>
struct ibeta_functor {
  NUMBIRCH_HOST_DEVICE T operator()(const T a, const T b, const T x) const {
    /* as of Eigen 3.4.0, the edge cases of a == 0 and b == 0 are not handled
    * internally, see https://gitlab.com/libeigen/eigen/-/issues/2359 */
    if (a == T(0) && b != T(0)) {
      return T(1);
    } else if (a != T(0) && b == T(0)) {
      return T(0);
    } else {
      return Eigen::numext::betainc(a, b, x);
    }
  }
};

}
