/**
 * @file
 */
#pragma once

#include "numbirch/random.hpp"
#include "numbirch/reduce.hpp"

namespace numbirch {

template<numeric T>
NUMBIRCH_KEEP bool_t<T> simulate_bernoulli(const T& rho) {
  return transform(rho, simulate_bernoulli_functor());
}

template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<T,U> simulate_beta(const T& alpha, const U& beta) {
  return transform(alpha, beta, simulate_beta_functor());
}

template<numeric T, numeric U>
NUMBIRCH_KEEP int_t<T,U> simulate_binomial(const T& n, const U& rho) {
  return transform(n, rho, simulate_binomial_functor());
}

template<numeric T>
NUMBIRCH_KEEP real_t<T> simulate_chi_squared(const T& nu) {
  return transform(nu, simulate_chi_squared_functor());
}

template<numeric T>
NUMBIRCH_KEEP real_t<T> simulate_dirichlet(const T& alpha) {
  auto x = simulate_gamma(alpha, real(1));
  return div(x, sum(x));
}

template<numeric T>
NUMBIRCH_KEEP real_t<T> simulate_exponential(const T& lambda) {
  return transform(lambda, simulate_exponential_functor());
}

template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<T,U> simulate_gamma(const T& k, const U& theta) {
  return transform(k, theta, simulate_gamma_functor());
}

template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<T,U> simulate_gaussian(const T& mu, const U& sigma2) {
  return transform(mu, sigma2, simulate_gaussian_functor());
}

template<numeric T, numeric U>
NUMBIRCH_KEEP int_t<T,U> simulate_negative_binomial(const T& k, const U& rho) {
  return transform(k, rho, simulate_negative_binomial_functor());
}

template<numeric T>
NUMBIRCH_KEEP int_t<T> simulate_poisson(const T& lambda) {
  return transform(lambda, simulate_poisson_functor());
}

template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<T,U> simulate_uniform(const T& l, const U& u) {
  return transform(l, u, simulate_uniform_functor());
}

template<numeric T, numeric U>
NUMBIRCH_KEEP int_t<T,U> simulate_uniform_int(const T& l, const U& u) {
  return transform(l, u, simulate_uniform_int_functor());
}

template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<T,U> simulate_weibull(const T& k, const U& lambda) {
  return transform(k, lambda, simulate_weibull_functor());
}

template<scalar T>
NUMBIRCH_KEEP Array<real,2> simulate_wishart(const T& nu, const int n) {
  Array<real,2> S(make_shape(n, n));
  for_each(n, n, simulate_wishart_functor(buffer(nu), n, buffer(S),
      stride(S)));
  return S;
}

NUMBIRCH_KEEP Array<real,1> standard_gaussian(const int n) {
  Array<real,1> x(make_shape(n));
  for_each(n, standard_gaussian_functor(buffer(x), stride(x)));
  return x;
}

NUMBIRCH_KEEP Array<real,2> standard_gaussian(const int m, const int n) {
  Array<real,2> X(make_shape(m, n));
  for_each(m, n, standard_gaussian_functor(buffer(X), stride(X)));
  return X;
}

}
