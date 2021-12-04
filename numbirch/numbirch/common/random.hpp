/**
 * @file
 */
#pragma once

#include "numbirch/random.hpp"
#include "numbirch/common/functor.hpp"

namespace numbirch {

template<class R, class T, class>
explicit_t<R,T> simulate_bernoulli(const T& ρ) {
  prefetch(ρ);
  return transform(ρ, simulate_bernoulli_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> simulate_beta(const T& α, const U& β) {
  prefetch(α);
  prefetch(β);
  return transform(α, β, simulate_beta_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> simulate_binomial(const T& n, const U& ρ) {
  prefetch(n);
  prefetch(ρ);
  return transform(n, ρ, simulate_binomial_functor<R>());
}

template<class R, class T, class>
explicit_t<R,T> simulate_chi_squared(const T& ν) {
  prefetch(ν);
  return transform(ν, simulate_chi_squared_functor<R>());
}

template<class R, class T, class>
explicit_t<R,T> simulate_exponential(const T& λ) {
  prefetch(λ);
  return transform(λ, simulate_exponential_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> simulate_gamma(const T& k, const U& θ) {
  prefetch(k);
  prefetch(θ);
  return transform(k, θ, simulate_gamma_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> simulate_gaussian(const T& μ, const U& σ2) {
  prefetch(μ);
  prefetch(σ2);
  return transform(μ, σ2, simulate_gaussian_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> simulate_negative_binomial(const T& k,
    const U& ρ) {
  prefetch(k);
  prefetch(ρ);
  return transform(k, ρ, simulate_negative_binomial_functor<R>());
}

template<class R, class T, class>
explicit_t<R,T> simulate_poisson(const T& λ) {
  prefetch(λ);
  return transform(λ, simulate_poisson_functor<R>());
}

template<class R, class T, class>
explicit_t<R,T> simulate_student_t(const T& ν) {
  prefetch(ν);
  return transform(ν, simulate_student_t_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> simulate_uniform(const T& l, const U& u) {
  prefetch(l);
  prefetch(u);
  return transform(l, u, simulate_uniform_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> simulate_uniform_int(const T& l, const U& u) {
  prefetch(l);
  prefetch(u);
  return transform(l, u, simulate_uniform_int_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> simulate_weibull(const T& k, const U& λ) {
  prefetch(k);
  prefetch(λ);
  return transform(k, λ, simulate_weibull_functor<R>());
}

}
