/**
 * @file
 */
#pragma once

#include "numbirch/random.hpp"
#include "numbirch/reduce.hpp"

namespace numbirch {

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
real_t<T> simulate_dirichlet(const T& alpha) {
  auto x = simulate_gamma(alpha, real(1));
  return div(x, sum(x));
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
Array<real,2> simulate_wishart(const T& nu, const int n) {
  Array<real,2> S(make_shape(n, n));
  for_each(n, n, simulate_wishart_functor(sliced(nu), n, sliced(S),
      stride(S)));
  return S;
}

}
