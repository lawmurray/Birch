/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

inline Real logz_conway_maxwell_poisson(const Real& μ, const Real& ν,
    const Integer& n) {
  /* to avoid taking exp() of large negative values, renormalize each term in
   * this sum using the maximum term, which is the unnormalized log
   * probability at the mode; this is similar to log_sum_exp() */
  auto log_λ = ν*log(μ);
  auto mode = std::min(μ, cast<Real>(n));
  auto mx = mode*log_λ - ν*lfact(mode);

  /* sum renormalized terms for x in 0..n */
  auto log_xf = Real(0.0);  // accumulator of log(x!)
  auto Z = exp(-(mx));  // x = 0 case
  for (int x = int(1); x <= int(n); ++x) {
    log_xf = log_xf + log(x);
    Z = Z + exp(x*log_λ - ν*log_xf - mx);
  }
  return mx + log(Z);
}

template<class Gradient, class Value>
inline auto logz_conway_maxwell_poisson_grad1(const Gradient& g,
    const Value& x, const Real& μ, const Real& ν, const Integer& n) {
  auto log_λ = ν*log(μ);
  auto mx = μ*log_λ - ν*lfact(μ);  // renormalizer
  auto log_xf = Real(0.0);  // accumulator of lfact(x)
  auto z = Real(0.0);
  auto Z = exp(-(mx));  // for x == 0
  auto gλ = Real(0.0);
  for (int x = int(1); x <= int(n); ++x) {
    log_xf = log_xf + log(x);
    z = exp(x*log_λ - ν*log_xf - mx);
    Z = Z + z;
    gλ = gλ + x*z;
  }
  return g*gλ*ν/(μ*Z);
}

template<class Gradient, class Value>
inline auto logz_conway_maxwell_poisson_grad2(const Gradient& g,
    const Value& x, const Real& μ, const Real& ν, const Integer& n) {
  auto log_λ = ν*log(μ);
  auto mx = μ*log_λ - ν*lfact(μ);  // renormalizer
  auto log_xf = Real(0.0);  // accumulator of lfact(x)
  auto z = Real(0.0);
  auto Z = exp(-(mx));  // for x == 0
  auto gλ = Real(0.0);
  auto gν = Real(0.0);
  for (int x = int(1); x <= int(n); ++x) {
    log_xf = log_xf + log(x);
    z = exp(x*log_λ - ν*log_xf - mx);
    Z = Z + z;
    gλ = gλ + x*z;
    gν = gν - log_xf*z;
  }
  return g*(gν + gλ*log(μ))/Z;
}

template<class Gradient, class Value>
inline auto logz_conway_maxwell_poisson_grad3(const Gradient& g,
    const Value& x, const Real& μ, const Real& ν, const Integer& n) {
  return Real(0.0);
}

template<class Left, class Middle, class Right>
struct LogZConwayMaxwellPoisson : public Ternary<Left,Middle,Right> {
  template<class T, class U, class V>
  LogZConwayMaxwellPoisson(T&& l, U&& m, V&& r) :
      Ternary<Left,Middle,Right>(std::forward<T>(l), std::forward<U>(m),
      std::forward<V>(r)) {
    //
  }

  BIRCH_TERNARY_FORM(logz_conway_maxwell_poisson)
  BIRCH_TERNARY_GRAD(logz_conway_maxwell_poisson_grad)
  BIRCH_FORM
  BIRCH_FORM_OP
};

/**
 * Logarithm of the normalizing constant of a Conway-Maxwell-Poisson
 * distribution truncated on a finite interval $[0,n]$.
 *
 * @param μ Mode.
 * @param ν Dispersion.
 * @param n Truncation point.
 *
 * @return vector of probabilities on $[0,n]$.
 */
template<class Left, class Middle, class Right, std::enable_if_t<
    is_delay_v<Left,Middle,Right>,int> = 0>
LogZConwayMaxwellPoisson<Left,Middle,Right> logz_conway_maxwell_poisson(
    const Left& λ, const Middle& ν, const Right& n) {
  return LogZConwayMaxwellPoisson<Left,Middle,Right>(λ, ν, n);
}

}
