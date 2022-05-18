/**
 * @file
 */
#pragma once

#include "numbirch/common/random.hpp"
#include "numbirch/eigen/transform.hpp"
#include "numbirch/random.hpp"
#include "numbirch/utility.hpp"

namespace numbirch {

struct simulate_bernoulli_functor {
  bool operator()(const real rho) const {
    return std::bernoulli_distribution(rho)(stl<bool>::rng());
  }
};

struct simulate_beta_functor {
  real operator()(const real alpha, const real beta) const {
    real u, v;
    auto& rng = stl<real>::rng();
    u = std::gamma_distribution<real>(alpha)(rng);
    v = std::gamma_distribution<real>(beta)(rng);
    return u/(u + v);
  }
};

struct simulate_binomial_functor {
  int operator()(const int n, const real rho) const {
    return std::binomial_distribution<int>(n, rho)(stl<int>::rng());
  }
};

struct simulate_chi_squared_functor {
  real operator()(const real nu) const {
    return std::chi_squared_distribution<real>(nu)(stl<real>::rng());
  }
};

struct simulate_exponential_functor {
  real operator()(const real lambda) const {
    return std::exponential_distribution<real>(lambda)(stl<real>::rng());
  }
};

struct simulate_gamma_functor {
  real operator()(const real k, const real theta) const {
    return std::gamma_distribution<real>(k, theta)(stl<real>::rng());
  }
};

struct simulate_gaussian_functor {
  real operator()(const real mu, const real sigma2) const {
    real sigma = std::sqrt(sigma2);
    return std::normal_distribution<real>(mu, sigma)(stl<real>::rng());
  }
};

struct simulate_negative_binomial_functor {
  int operator()(const int k, const real rho) const {
    return std::negative_binomial_distribution<int>(k, rho)(stl<int>::rng());
  }
};

struct simulate_poisson_functor {
  int operator()(const real lambda) const {
    return std::poisson_distribution<int>(lambda)(stl<int>::rng());
  }
};

struct simulate_uniform_functor {
  real operator()(const real l, const real u) const {
    return std::uniform_real_distribution<real>(l, u)(stl<real>::rng());
  }
};

struct simulate_uniform_int_functor {
  int operator()(const int l, const int u) const {
    return std::uniform_int_distribution<int>(l, u)(stl<int>::rng());
  }
};

struct simulate_weibull_functor {
  real operator()(const real k, const real lambda) const {
    return std::weibull_distribution<real>(k, lambda)(stl<real>::rng());
  }
};

struct standard_gaussian_functor {
  real operator()(const int i, const int j) const {
    return std::normal_distribution<real>()(stl<real>::rng());
  }
};

template<class T>
struct standard_wishart_functor {
  T k;
  int n;
  standard_wishart_functor(const T& k, const int n) :
      k(k),
      n(n) {
    //
  }
  real operator()(const int i, const int j) {
    auto& rng = stl<real>::rng();
    if (i == j) {
      /* on diagonal */
      real nu = get(k) + (n - 1 - i); // i is 0-based here
      real x = std::chi_squared_distribution<real>(nu)(rng);
      return std::sqrt(x);
    } else if (i > j) {
      /* in lower triangle */
      return std::normal_distribution<real>()(rng);
    } else {
      /* in upper triangle */
      return real(0.0);
    }
  }
};

template<class T, class>
bool_t<T> simulate_bernoulli(const T& rho) {
  prefetch(rho);
  return transform(rho, simulate_bernoulli_functor());
}

template<class T, class U, class>
real_t<T,U> simulate_beta(const T& alpha, const U& beta) {
  prefetch(alpha);
  prefetch(beta);
  return transform(alpha, beta, simulate_beta_functor());
}

template<class T, class U, class>
int_t<T,U> simulate_binomial(const T& n, const U& rho) {
  prefetch(n);
  prefetch(rho);
  return transform(n, rho, simulate_binomial_functor());
}

template<class T, class>
real_t<T> simulate_chi_squared(const T& nu) {
  prefetch(nu);
  return transform(nu, simulate_chi_squared_functor());
}

template<class T, class>
real_t<T> simulate_exponential(const T& lambda) {
  prefetch(lambda);
  return transform(lambda, simulate_exponential_functor());
}

template<class T, class U, class>
real_t<T,U> simulate_gamma(const T& k, const U& theta) {
  prefetch(k);
  prefetch(theta);
  return transform(k, theta, simulate_gamma_functor());
}

template<class T, class U, class>
real_t<T,U> simulate_gaussian(const T& mu, const U& sigma2) {
  prefetch(mu);
  prefetch(sigma2);
  return transform(mu, sigma2, simulate_gaussian_functor());
}

template<class T, class U, class>
int_t<T,U> simulate_negative_binomial(const T& k, const U& rho) {
  prefetch(k);
  prefetch(rho);
  return transform(k, rho, simulate_negative_binomial_functor());
}

template<class T, class>
int_t<T> simulate_poisson(const T& lambda) {
  prefetch(lambda);
  return transform(lambda, simulate_poisson_functor());
}

template<class T, class U, class>
real_t<T,U> simulate_uniform(const T& l, const U& u) {
  prefetch(l);
  prefetch(u);
  return transform(l, u, simulate_uniform_functor());
}

template<class T, class U, class>
int_t<T,U> simulate_uniform_int(const T& l, const U& u) {
  prefetch(l);
  prefetch(u);
  return transform(l, u, simulate_uniform_int_functor());
}

template<class T, class U, class>
real_t<T,U> simulate_weibull(const T& k, const U& lambda) {
  prefetch(k);
  prefetch(lambda);
  return transform(k, lambda, simulate_weibull_functor());
}

template<class T, class>
Array<real,2> standard_wishart(const T& nu, const int n) {
  return for_each(n, n, standard_wishart_functor<decltype(data(nu))>(
      data(nu), n));
}

}