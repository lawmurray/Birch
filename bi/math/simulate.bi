/**
 * Seed the pseudorandom number generator.
 *
 * - seed: Seed value.
 */
function seed(s:Integer) {
  cpp {{
  rng.seed(s_);
  }}
}

/**
 * Simulate a Bernoulli variate.
 *
 * - ρ: Probability of a true result.
 */
function simulate_bernoulli(ρ:Real) -> Boolean {
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp {{
  return std::bernoulli_distribution(ρ_)(rng);
  }}
}

/**
 * Simulate an integer uniform variate.
 *
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 */
function simulate_int_uniform(l:Integer, u:Integer) -> Integer {
  assert l <= u;
  cpp {{
  return std::uniform_int_distribution<bi::Integer_>(l_, u_)(rng);
  }}
}

/**
 * Simulate a binomial variate.
 *
 * - n: Number of trials.
 * - ρ: Probability of a true result.
 */
function simulate_binomial(n:Integer, ρ:Real) -> Integer {
  assert 0 <= n;
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp {{
  return std::binomial_distribution<bi::Integer_>(n_, ρ_)(rng);
  }}
}

/**
 * Simulate a negative binomial variate.
 *
 * - k: Number of successes before the experiment is stopped.
 * - ρ: Probability of success.
 *
 * Returns the number of failures.
 */
function simulate_negative_binomial(k:Integer, ρ:Real) -> Integer {
  assert 0 < k;
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp {{
  return std::negative_binomial_distribution<bi::Integer_>(k_, ρ_)(rng);
  }}
}

/**
 * Simulate a beta-binomial variate.
 *
 * - n: Number of trials.
 * - α: Shape.
 * - β: Shape.
 */
function simulate_beta_binomial(n:Integer, α:Real, β:Real) -> Integer {
  assert 0 <= n;
  assert 0.0 < α;
  assert 0.0 < β;
  
  return simulate_binomial(n, simulate_beta(α, β));
}

/**
 * Simulate a Poisson variate.
 *
 * - λ: Rate.
 */
function simulate_poisson(λ:Real) -> Integer {
  assert 0.0 <= λ;
  if (λ > 0.0) {
    cpp {{
    return std::poisson_distribution<bi::Integer_>(λ_)(rng);
    }}
  } else {
    return 0;
  }
}

/**
 * Simulate a uniform variate.
 *
 * - l: Lower bound of interval.
 * - u: Upper bound of interval.
 */
function simulate_uniform(l:Real, u:Real) -> Real {
  assert l <= u;
  cpp {{
  return std::uniform_real_distribution<bi::Real_>(l_, u_)(rng);
  }}
}

/**
 * Simulate an exponential variate.
 *
 * - λ: Rate.
 */
function simulate_exponential(λ:Real) -> Real {
  assert 0.0 <= λ;
  cpp {{
  return std::exponential_distribution<bi::Real_>(λ_)(rng);
  }}
}

/**
 * Simulate a Gaussian variate.
 *
 * - μ: Mean.
 * - σ2: Variance.
 */
function simulate_gaussian(μ:Real, σ2:Real) -> Real {
  assert 0.0 <= σ2;
  if (σ2 == 0.0) {
    return μ;
  } else {
    cpp {{
    return std::normal_distribution<bi::Real_>(μ_, ::sqrt(σ2_))(rng);
    }}
  }
}

/**
 * Simulate a log-Gaussian variate.
 *
 * - μ: Mean (in log space).
 * - σ2: Variance (in log space).
 */
function simulate_log_gaussian(μ:Real, σ2:Real) -> Real {
  assert 0.0 <= σ2;
  if (σ2 == 0.0) {
    return μ;
  } else {
    cpp {{
    return std::lognormal_distribution<bi::Real_>(μ_, ::sqrt(σ2_))(rng);
    }}
  }
}

/**
 * Simulate a gamma variate.
 *
 * - k: Shape.
 * - θ: Scale.
 */
function simulate_gamma(k:Real, θ:Real) -> Real {
  assert 0.0 < k;
  assert 0.0 < θ;
  cpp {{
  return std::gamma_distribution<bi::Real_>(k_, θ_)(rng);
  }}
}

/**
 * Simulate a beta variate.
 *
 * - α: Shape.
 * - β: Shape.
 */
function simulate_beta(α:Real, β:Real) -> Real {
  assert 0.0 < α;
  assert 0.0 < β;
  
  u:Real <- simulate_gamma(α, 1.0);
  v:Real <- simulate_gamma(β, 1.0);
  
  return u/(u + v);
}

/**
 * Simulate a categorical variate.
 *
 * - ρ: Category probabilities.
 */
function simulate_categorical(ρ:Real[_]) -> Integer {
  /* assertion checks throughout catch cases such as negative probabilities,
   * or the sum of probabilities not being one */
    
  u:Real <- simulate_uniform(0.0, 1.0);
  x:Integer <- 1;
  assert length(ρ) > 0;
  P:Real <- ρ[1];
  assert 0.0 <= P && P <= 1.0;
  while (u < P) {
    x <- x + 1;
    assert x <= length(ρ);
    assert ρ[x] >= 0.0;
    P <- P + ρ[x];
    assert 0.0 <= P && P <= 1.0;
  }
  return x;
}

/**
 * Simulate a multinomial variate.
 *
 * - n: Number of trials.
 * - ρ: Category probabilities.
 */
function simulate_multinomial(n:Integer, ρ:Real[_]) -> Integer[_] {
  D:Integer <- length(ρ);
  x:Integer[_] <- vector(0, D);
  R:Integer[_] <- inclusive_prefix_sum(ρ);
  
  i:Integer <- n;
  j:Integer <- D;
  mx:Real <- 0.0;
  u:Real;
  
  for (i in 1..n) {
    mx <- mx + log(simulate_uniform(0.0, 1.0))/(n - i + 1);
    u <- 1.0 + mx;
    while (j >= 1 && u < log(R[j])) {
      j <- j - 1;
    }
    x[j] <- x[j] + 1;
  }
  return x;
}

/**
 * Simulate a Dirichlet variate.
 *
 * - α: Concentrations.
 */
function simulate_dirichlet(α:Real[_]) -> Real[_] {
  D:Integer <- length(α);
  x:Real[D];
  z:Real <- 0.0;

  for (i:Integer in 1..D) {
    x[i] <- simulate_gamma(α[i], 1.0);
    z <- z + x[i];
  }
  z <- 1.0/z;
  for (i:Integer in 1..D) {
    x[i] <- z*x[i];
  }
  return x;
}
