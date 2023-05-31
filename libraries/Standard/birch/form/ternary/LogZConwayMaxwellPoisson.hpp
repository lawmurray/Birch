/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace numbirch {
inline real logz_conway_maxwell_poisson(const real& μ, const real& ν,
    const int& n) {
  /* to avoid taking exp() of large negative values, renormalize each term in
   * this sum using the maximum term, which is the unnormalized log
   * probability at the mode; this is similar to log_sum_exp() */
  auto log_λ = ν*log(μ);
  auto mode = std::min(μ, cast<real>(n));
  auto mx = mode*log_λ - ν*lfact(mode);

  /* sum renormalized terms for x in 0..n */
  auto log_xf = real(0.0);  // accumulator of log(x!)
  auto Z = exp(-(mx));  // x = 0 case
  for (int x = int(1); x <= int(n); ++x) {
    log_xf = log_xf + log(x);
    Z = Z + exp(x*log_λ - ν*log_xf - mx);
  }
  return mx + log(Z);
}

template<class Gradient, class Value>
inline auto logz_conway_maxwell_poisson_grad1(const Gradient& g,
    const Value& x, const real& μ, const real& ν, const int& n) {
  auto log_λ = ν*log(μ);
  auto mx = μ*log_λ - ν*lfact(μ);  // renormalizer
  auto log_xf = real(0.0);  // accumulator of lfact(x)
  auto z = real(0.0);
  auto Z = exp(-(mx));  // for x == 0
  auto gλ = real(0.0);
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
    const Value& x, const real& μ, const real& ν, const int& n) {
  auto log_λ = ν*log(μ);
  auto mx = μ*log_λ - ν*lfact(μ);  // renormalizer
  auto log_xf = real(0.0);  // accumulator of lfact(x)
  auto z = real(0.0);
  auto Z = exp(-(mx));  // for x == 0
  auto gλ = real(0.0);
  auto gν = real(0.0);
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
    const Value& x, const real& μ, const real& ν, const int& n) {
  return real(0.0);
}

}

namespace birch {

template<class Left, class Middle, class Right>
struct LogZConwayMaxwellPoisson {
  BIRCH_TERNARY_FORM(LogZConwayMaxwellPoisson, numbirch::logz_conway_maxwell_poisson)
  BIRCH_TERNARY_GRAD(numbirch::logz_conway_maxwell_poisson_grad)
  BIRCH_FORM
};

/**
 * Logarithm of the normalizing constant of a Conway-Maxwell-Poisson
 * distribution truncated on a finite interval $[0,n]$.
 *
 * @param l Mode.
 * @param m Dispersion.
 * @param r Truncation point.
 *
 * @return Logarithm of normalizing constant.
 */
template<class Left, class Middle, class Right>
auto logz_conway_maxwell_poisson(const Left& l, const Middle& m,
    const Right& r) {
  if constexpr (numbirch::is_arithmetic_v<Left> &&
      numbirch::is_arithmetic_v<Middle> &&
      numbirch::is_arithmetic_v<Right>) {
    return numbirch::logz_conway_maxwell_poisson(l, m, r);
  } else {
    return BIRCH_TERNARY_CONSTRUCT(LogZConwayMaxwellPoisson);
  }
}

}
