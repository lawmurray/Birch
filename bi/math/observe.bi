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
 * Observe a binomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - ρ: Probability of a true result.
 
 * Returns the log probability mass.
 */
function observe_binomial(x:Integer, n:Integer, ρ:Real) -> Real {
  assert 0 <= n;
  assert 0.0 <= ρ && ρ <= 1.0;

  if (0 <= x && x <= n) {
    return x*log(ρ) + (n - x)*log(1.0 - ρ) + lchoose(n, x);
  } else {
    return -inf;
  }
}

/**
 * Observe a negative binomial variate.
 *
 * - x: The variate (number of failures).
 * - k: Number of successes before the experiment is stopped.
 * - ρ: Probability of success.
 *
 * Returns the log probability mass.
 */
function observe_negative_binomial(x:Integer, k:Integer, ρ:Real) -> Real {
  assert 0 < k;
  assert 0.0 <= ρ && ρ <= 1.0;

  if (x >= 0) {
    return k*log(ρ) + x*log(1.0 - ρ) + lchoose(x + k - 1, x);
  } else {
    return -inf;
  }
}

/**
 * Observe a beta-binomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - α: Shape.
 * - β: Shape.
 *
 * Returns the log probability density.
 *
 * Returns the log probability mass.
 */
function observe_beta_binomial(x:Integer, n:Integer, α:Real, β:Real) -> Real {
  assert 0 <= n;
  assert 0.0 < α;
  assert 0.0 < β;

  if (0 <= x && x <= n) {
    return lbeta(x + α, n - x + β) - lbeta(α, β) + lchoose(n, x);
  } else {
    return -inf;
  }
}

/**
 * Observe a Poisson variate.
 *
 * - x: The variate.
 * - λ: Rate.
 *
 * Returns the log probability mass.
 */
function observe_poisson(x:Integer, λ:Real) -> Real {
  assert 0.0 <= λ;

  if (λ > 0.0) {
    if (x >= 0) {
      return x*log(λ) - λ - lgamma(x + 1);
    } else {
      return -inf;
    }
  } else {
    if (x == 0) {
      return inf;
    } else {
      return -inf;
    }
  }
}

/**
 * Observe a uniform variate.
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
  assert 0.0 <= σ2;
  
  if (σ2 == 0.0) {
    if (x == μ) {
      return inf;
    } else {
      return -inf;
    }
  } else {
    return -0.5*(pow(x - μ, 2.0)/σ2 + log(2.0*π*σ2));
  }
}

/**
 * Observe a log-Gaussian variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Returns the log probability density.
 */
function observe_log_gaussian(x:Real, μ:Real, σ2:Real) -> Real {
  if (x > 0.0) {
    return observe_log_gaussian(log(x), μ, σ2) - log(x);
  } else {
    return -inf;
  }
}

/**
 * Observe a gamma variate.
 *
 * - x: The variate.
 * - k: Shape.
 * - θ: Scale.
 *
 * Returns the log probability density.
 */
function observe_gamma(x:Real, k:Real, θ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < θ;
  
  if (x > 0.0) {
    return (k - 1.0)*log(x) - x/θ - lgamma(k) - k*log(θ);
  } else {
    return -inf;
  }
}

/**
 * Observe a beta variate.
 *
 * - x: The variate.
 * - α: Shape.
 * - β: Shape.
 *
 * Returns the log probability density.
 */
function observe_beta(x:Real, α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;

  if (0.0 < x && x < 1.0) {
    return (α - 1.0)*log(x) + (β - 1.0)*log(1.0 - x) - lbeta(α, β);
  } else {
    return -inf;
  }
}
