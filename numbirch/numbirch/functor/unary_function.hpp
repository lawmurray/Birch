/**
 * @file
 */
#pragma once

#include "numbirch/functor/macro.hpp"
#include "numbirch/functor/function.hpp"

namespace numbirch {

template<class T>
struct abs_functor {
  DEVICE T operator()(const T x) const {
    return std::abs(x);
  }
};

template<class T>
struct acos_functor {
  DEVICE T operator()(const T x) const {
    return std::acos(x);
  }
};

template<class T>
struct asin_functor {
  DEVICE T operator()(const T x) const {
    return std::asin(x);
  }
};

template<class T>
struct atan_functor {
  DEVICE T operator()(const T x) const {
    return std::atan(x);
  }
};

template<class T>
struct ceil_functor {
  DEVICE T operator()(const T x) const {
    return std::ceil(x);
  }
};

template<class T>
struct cos_functor {
  DEVICE T operator()(const T x) const {
    return std::cos(x);
  }
};

template<class T>
struct cosh_functor {
  DEVICE T operator()(const T x) const {
    return std::cosh(x);
  }
};

template<class T>
struct digamma_functor {
  DEVICE T operator()(const T x) const {
    return digamma(x);
  }
};

template<class T>
struct exp_functor {
  DEVICE T operator()(const T x) const {
    return std::exp(x);
  }
};

template<class T>
struct expm1_functor {
  DEVICE T operator()(const T x) const {
    return std::expm1(x);
  }
};

template<class T>
struct floor_functor {
  DEVICE T operator()(const T x) const {
    return std::floor(x);
  }
};

template<class T>
struct lgamma_functor {
  DEVICE T operator()(const T x) const {
    return std::lgamma(x);
  }
};

template<class T>
struct log_functor {
  DEVICE T operator()(const T x) const {
    return std::log(x);
  }
};

template<class T>
struct log1p_functor {
  DEVICE T operator()(const T x) const {
    return std::log1p(x);
  }
};

template<class T>
struct log_abs_functor {
  DEVICE T operator()(const T x) const {
    return std::log(std::abs(x));
  }
};

template<class T>
struct log_square_functor {
  DEVICE T operator()(const T x) const {
    return 2.0*std::log(x);
  }
};

template<class T>
struct rectify_functor {
  DEVICE T operator()(const T x) const {
    return x > T(0) ? x : T(0);
  }
};

template<class T>
struct round_functor {
  DEVICE T operator()(const T x) const {
    return std::round(x);
  }
};

template<class T>
struct sin_functor {
  DEVICE T operator()(const T x) const {
    return std::sin(x);
  }
};

template<class T>
struct sinh_functor {
  DEVICE T operator()(const T x) const {
    return std::sinh(x);
  }
};

template<class T>
struct sqrt_functor {
  DEVICE T operator()(const T x) const {
    return std::sqrt(x);
  }
};

template<class T>
struct tan_functor {
  DEVICE T operator()(const T x) const {
    return std::tan(x);
  }
};

template<class T>
struct tanh_functor {
  DEVICE T operator()(const T x) const {
    return std::tanh(x);
  }
};

}
