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
 * Simulate a Poisson variate.
 *
 * - λ: Rate.
 */
function simulate_poisson(λ:Real) -> Integer {
  assert 0.0 < λ;
  cpp {{
  return std::poisson_distribution<bi::Integer_>(λ_)(rng);
  }}
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
