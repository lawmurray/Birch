cpp{{
#include "boost/math/distributions.hpp"
}}

/**
 * CDF of a binomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - ρ: Probability of a true result.
 *
 * Returns: the cumulative probability.
 */
function cdf_binomial(x:Integer, n:Integer, ρ:Real) -> Real {
  assert 0 <= n;
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp{{
  return boost::math::cdf(boost::math::binomial_distribution<>(n_, ρ_), x_);
  }}
}

/**
 * CDF of a negative binomial variate.
 *
 * - x: The variate (number of failures).
 * - k: Number of successes before the experiment is stopped.
 * - ρ: Probability of success.
 *
 * Returns: the cumulative probability.
 */
function cdf_negative_binomial(x:Integer, k:Integer, ρ:Real) -> Real {
  assert 0 < k;
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp{{
  return boost::math::cdf(boost::math::negative_binomial_distribution<>(k_, ρ_), x_);
  }}
}

/**
 * CDF of a Poisson variate.
 *
 * - x: The variate.
 * - λ: Rate.
 *
 * Returns: the cumulative probability.
 */
function cdf_poisson(x:Integer, λ:Real) -> Real {
  assert 0.0 <= λ;
  cpp{{
  return boost::math::cdf(boost::math::poisson_distribution<>(λ_), x_);
  }}
}

/**
 * CDF of a uniform variate.
 *
 * - x: The variate.
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 *
 * Returns: the cumulative probability.
 */
function cdf_uniform(x:Real, l:Real, u:Real) -> Real {
  assert l <= u;
  return (x - l)/(u - l);
}

/**
 * CDF of an exponential variate.
 *
 * - x: The variate.
 * - λ: Rate.
 *
 * Returns: the cumulative probability.
 */
function cdf_exponential(x:Real, λ:Real) -> Real {
  assert 0.0 < λ;
  cpp{{
  return boost::math::cdf(boost::math::exponential_distribution<>(λ_), x_);
  }}
}

/**
 * CDF of a Gaussian variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Returns: the cumulative probability.
 */
function cdf_gaussian(x:Real, μ:Real, σ2:Real) -> Real {
  assert 0.0 < σ2;
  cpp{{
  return boost::math::cdf(boost::math::normal_distribution<>(μ_, ::sqrt(σ2_)), x_);
  }}
}

/**
 * CDF of a log-Gaussian variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Returns: the cumulative probability.
 */
function cdf_log_gaussian(x:Real, μ:Real, σ2:Real) -> Real {
  assert 0.0 < σ2;
  cpp{{
  return boost::math::cdf(boost::math::lognormal_distribution<>(μ_, ::sqrt(σ2_)), x_);
  }}
}

/**
 * CDF of a Student's $t$ variate.
 *
 * - x: The variate.
 * - ν: Degrees of freedom.
 *
 * Returns: the cumulative probability.
 */
function cdf_student_t(x:Real, ν:Real) -> Real {
  assert 0.0 < ν;
  cpp{{
  return boost::math::cdf(boost::math::students_t_distribution<>(ν_), x_);
  }}
}

/**
 * CDF of a Student's $t$ variate with location and scale.
 *
 * - x: The variate.
 * - ν: Degrees of freedom.
 * - μ: Location.
 * - σ2: Squared scale.
 *
 * Returns: the cumulative probability.
 */
function cdf_student_t(x:Real, ν:Real, μ:Real, σ2:Real) -> Real {
  assert 0.0 < σ2;
  return cdf_student_t((x - μ)/sqrt(σ2), ν);
}

/**
 * CDF of a beta variate.
 *
 * - x: The variate.
 * - α: Shape.
 * - β: Shape.
 *
 * Returns: the cumulative probability.
 */
function cdf_beta(x:Real, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;
  cpp{{
  return boost::math::cdf(boost::math::beta_distribution<>(α_, β_), x_);
  }}
}

/**
 * CDF of a gamma variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - θ: Scale.
 *
 * Returns: the cumulative probability.
 */
function cdf_gamma(x:Real, k:Real, θ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < θ;
  cpp{{
  return boost::math::cdf(boost::math::gamma_distribution<>(k_, θ_), x_);
  }}
}

/**
 * CDF of an inverse-gamma variate.
 *
 * - x: The variate.
 * - α: Shape.
 * - β: Scale.
 *
 * Returns: the cumulative probability.
 */
function cdf_inverse_gamma(x:Real, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;
  cpp{{
  return boost::math::cdf(boost::math::inverse_gamma_distribution<>(α_, β_), x_);
  }}
}

/**
 * CDF of a normal inverse-gamma variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - a2: Variance.
 * - α: Shape of inverse-gamma on scale.
 * - β: Scale of inverse-gamma on scale.
 *
 * Returns: the cumulative probability.
 */
function cdf_normal_inverse_gamma(x:Real, μ:Real, a2:Real, α:Real,
    β:Real) -> Real {
  return cdf_student_t(x, 2.0*α, μ, a2*β/α);
}

/**
 * CDF of a beta-binomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - α: Shape.
 * - β: Shape.
 *
 * Returns: the cumulative probability.
 */
function cdf_beta_binomial(x:Integer, n:Integer, α:Real, β:Real) -> Real {
  P:Real <- 0.0;
  for (i:Integer in 0..x) {
    P <- P + exp(observe_beta_binomial(x, n, α, β));
  }
  return P;
}

/**
 * CDF of a gamma-Poisson variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - θ: Scale.
 *
 * Returns: the cumulative probability.
 */
function cdf_gamma_poisson(x:Integer, k:Integer, θ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < θ;
  return cdf_negative_binomial(x, k, 1.0/(θ + 1.0));
}

/**
 * CDF of a Gaussian variate with a normal inverse-gamma prior.
 *
 * - x: The variate.
 * - μ: Mean.
 * - a2: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Returns: the cumulative probability.
 */
function cdf_normal_inverse_gamma_gaussian(x:Real, μ:Real, a2:Real,
    α:Real, β:Real) -> Real {
  return cdf_student_t(x, 2.0*α, μ, (β/α)*(1.0 + a2));
}
