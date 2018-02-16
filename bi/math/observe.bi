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
 * Observe an integer uniform variate.
 *
 * - x: The variate.
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 *
 * Returns the log probability mass.
 */
function observe_int_uniform(x:Integer, l:Integer, u:Integer) -> Real {
  if (x >= l && x <= u) {
    return -log(u - l + 1);
  } else {
    return -inf;
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
 * Observe a categorical variate.
 *
 * - x: The variate.
 * - ρ: Category probabilities.
 *
 * Returns the log probability mass.
 */
function observe_categorical(x:Integer, ρ:Real[_]) -> Real {
  if (1 <= x && x <= length(ρ)) {
    assert ρ[x] >= 0.0;
    return log(ρ[x]);
  } else {
    return -inf;
  }
}

/**
 * Observe a multinomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - ρ: Category probabilities.
 *
 * Returns the log probability mass.
 */
function observe_multinomial(x:Integer[_], n:Integer, ρ:Real[_]) -> Real {
  assert length(x) == length(ρ);

  m:Integer <- 0;
  w:Real <- lgamma(n + 1.0);
  for (i:Integer in 1..length(x)) {
    assert x[i] >= 0;
    assert ρ[i] >= 0.0;
    m <- m + x[i];
    w <- w + x[i]*log(ρ[i]) - lgamma(x[i] + 1.0);
  }
  if (m == n) {
    return w;
  } else {
    return -inf;
  }
}

/**
 * Observe a Dirichlet-categorical variate.
 *
 * - x: The variate.
 * - α: Concentrations.
 *
 * Returns the log probability mass.
 */
function observe_dirichlet_categorical(x:Integer, α:Real[_]) -> Real {
  if (1 <= x && x <= length(α)) {
    A:Real <- sum(α);
    return lgamma(1.0 + α[x]) - lgamma(α[x]) + lgamma(A) - lgamma(1.0 + A);
  } else {
    return -inf;
  }
}

/**
 * Observe a Dirichlet-multinomial variate.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - α: Concentrations.
 *
 * Returns the log probability mass.
 */
function observe_dirichlet_multinomial(x:Integer[_], n:Integer, α:Real[_]) -> Real {
  assert length(x) == length(α);

  A:Real <- sum(α);
  m:Integer <- 0;
  w:Real <- lgamma(n + 1.0) + lgamma(A) - lgamma(n + A);
  for (i:Integer in 1..length(α)) {
    assert x[i] >= 0;
    m <- m + x[i];
    w <- w + lgamma(x[i] + α[i]) - lgamma(x[i] + 1.0) - lgamma(α[i]);
  }
  if (m == n) {
    return w;
  } else {
    return -inf;
  }
}

/**
 * Observe a categorical variate with Chinese restaurant process prior.
 */
function observe_crp_categorical(k:Integer, α:Real, θ:Real, n:Integer[_], N:Integer) -> Real {
  K:Integer <- length(n);
  if (k > K + 1) {
    return -inf;
  } else if (k == K + 1) {
    return (K*α + θ)/(N + θ);
  } else {
    return (n[k] - α)/(N + θ);
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
 * Observe a multivariate Gaussian variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - Σ: Covariance.
 *
 * Returns the log probability density.
 */
function observe_multivariate_gaussian(x:Real[_], μ:Real[_], Σ:Real[_,_]) -> Real {
  D:Integer <- length(μ);
  L:Real[_,_] <- llt(Σ);
  
  return -0.5*dot(solve(L, x - μ)) - log(determinant(L)) - 0.5*D*log(2.0*π);
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

/**
 * Observe a Dirichlet variate.
 *
 * - x: The variate.
 * - α: Concentrations.
 *
 * Returns the log probability density.
 */
function observe_dirichlet(x:Real[_], α:Real[_]) -> Real {
  assert length(x) == length(α);

  D:Integer <- length(x);
  w:Real <- 0.0;
  for (i:Integer in 1..D) {
    assert x[i] >= 0.0;
    w <- w + (α[i] - 1.0)*log(x[i]) - lgamma(α[i]);
  }
  w <- w + lgamma(sum(α)); 
  return w;
}
