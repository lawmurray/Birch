import math;

/**
 * Observe a Bernoulli variate.
 *
 * - x: The variate.
 * - ρ: Probability of a true result.
 *
 * Returns the log probability mass.
 */
function observe_bernoulli(x:Boolean, ρ:Real) -> Real {
  assert 0.0 <= ρ && ρ <= 1.0;
  if (x) {
    return log(ρ);
  } else {
    return log(1.0 - ρ);
  }
}

/**
 * Observe a Binomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - ρ: Probability of a true result.
 
 * Returns the log probability mass.
 */
function observe_binomial(x:Integer, n:Integer, ρ:Real) -> Real {
  assert 0 <= n;
  assert 0.0 <= ρ && ρ <= 1.0;
  assert 0 <= x && x <= n;

  return Real(x)*log(ρ) + Real(n - x)*log(1.0 - ρ) - log(Real(x)) - lbeta(Real(x), Real(n - x + 1));
}

/**
 * Observe a Uniform variate.
 *
 * - x: The variate.
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 *
 * Returns the log probability density.
 */
function observe_uniform(x:Real, l:Real, u:Real) -> Real {
  assert l <= u;

  if (x >= l && x <= u) {
    return -log(u - l);
  } else {
    return -inf;
  }
}

/**
 * Observe a Gaussian variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Returns the log probability density.
 */
function observe_gaussian(x:Real, μ:Real, σ2:Real) -> Real {
  assert σ2 >= 0.0;
  if (σ2 == 0.0) {
    return inf;
  } else {
    return -0.5*(pow(x - μ, 2.0)/σ2 + log(2.0*π*σ2));
  }
}

/**
 * Observe a Gamma variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - θ: Scale.
 *
 * Returns the log probability density.
 */
function observe_gamma(x:Real, k:Real, θ:Real) -> Real {
  assert k > 0.0;
  assert θ > 0.0;
  
  if (x > 0.0) {
    return (k - 1.0)*log(x) - x/θ - lgamma(k) - k*log(θ);
  } else {
    return -inf;
  }
}

/**
 * Observe a Beta variate.
 *
 * - x: The variate.
 * - α: Shape.
 * - β: Shape.
 *
 * Returns the log probability density.
 */
function observe_beta(x:Real, α:Real, β:Real) -> Real {
  assert α > 0.0;
  assert β > 0.0;

  if (0.0 < x && x < 1.0) {
    return (α - 1.0)*log(x) + (β - 1.0)*log(1.0 - x) - lbeta(α, β);
  } else {
    return -inf;
  }
}
