/**
 * @file
 */
#pragma once

#include "numbirch/random.hpp"
#include "numbirch/common/stl.hpp"
#include "numbirch/common/element.hpp"

namespace numbirch {

template<class R>
struct simulate_bernoulli_functor {
  template<class T>
  R operator()(const T rho) const {
    return std::bernoulli_distribution(rho)(stl<bool>::rng());
  }
};

template<class R>
struct simulate_beta_functor {
  template<class T, class U>
  R operator()(const T alpha, const U beta) const {
    auto u = std::gamma_distribution<R>(alpha, 1.0)(stl<R>::rng());
    auto v = std::gamma_distribution<R>(beta, 1.0)(stl<R>::rng());
    return u/(u + v);
  }
};

template<class R>
struct simulate_binomial_functor {
  template<class T, class U>
  R operator()(const T n, const U rho) const {
    return std::binomial_distribution<int>(n, rho)(stl<R>::rng());
    // ^ binomial_distribution requires integral type
  }
};

template<class R>
struct simulate_chi_squared_functor {
  template<class T>
  R operator()(const T nu) const {
    return std::chi_squared_distribution<R>(nu)(stl<R>::rng());
  }
};

template<class R>
struct simulate_exponential_functor {
  template<class T>
  R operator()(const T lambda) const {
    return std::exponential_distribution<R>(lambda)(stl<R>::rng());
  }
};

template<class R>
struct simulate_gamma_functor {
  template<class T, class U>
  R operator()(const T k, const U theta) const {
    return std::gamma_distribution<long double>(k, theta)(stl<R>::rng());
  }
};

template<class R>
struct simulate_gaussian_functor {
  template<class T, class U>
  R operator()(const T mu, const U sigma2) const {
    return std::normal_distribution<R>(mu, std::sqrt(sigma2))(stl<R>::rng());
  }
};

template<class R>
struct simulate_negative_binomial_functor {
  template<class T, class U>
  R operator()(const T k, const U rho) const {
    return std::negative_binomial_distribution<int>(k, rho)(stl<R>::rng());
    // ^ negative_binomial_distribution requires integral type
  }
};

template<class R>
struct simulate_poisson_functor {
  template<class T>
  R operator()(const T lambda) const {
    return std::poisson_distribution<int>(lambda)(stl<R>::rng());
    // ^ poisson_distribution requires integral type
  }
};

template<class R>
struct simulate_student_t_functor {
  template<class T>
  R operator()(const T nu) const {
    return std::student_t_distribution<R>(nu)(stl<R>::rng());
  }
};

template<class R>
struct simulate_uniform_functor {
  template<class T, class U>
  R operator()(const T l, const U u) const {
    return std::uniform_real_distribution<R>(l, u)(stl<R>::rng());
  }
};

template<class R>
struct simulate_uniform_int_functor {
  template<class T, class U>
  R operator()(const T l, const U u) const {
    return std::uniform_int_distribution<int>(l, u)(stl<R>::rng());
    // ^ uniform_int_distribution requires integral type
  }
};

template<class R>
struct simulate_weibull_functor {
  template<class T, class U>
  R operator()(const T k, const U lambda) const {
    return std::weibull_distribution<R>(k, lambda)(stl<R>::rng());
  }
};

template<class R>
struct standard_gaussian_functor {
  NUMBIRCH_HOST_DEVICE R operator()(const int i, const int j) const {
    return std::normal_distribution<R>()(stl<R>::rng());
  }
};

template<class R, class T>
struct standard_wishart_functor {
  T k;
  int n;
  standard_wishart_functor(const T& k, const int n) :
      k(k),
      n(n) {
    //
  }
  NUMBIRCH_HOST_DEVICE R operator()(const int i, const int j) {
    if (i == j) {
      /* on diagonal */
      R nu = element(k) + n - i;
      R x = std::chi_squared_distribution<R>(nu)(stl<R>::rng());
      return std::sqrt(x);
    } else if (i > j) {
      /* in lower triangle */
      return std::normal_distribution<R>()(stl<R>::rng());
    } else {
      /* in upper triangle */
      return R(0.0);
    }
  }
};
}