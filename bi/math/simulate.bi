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
  return std::uniform_int_distribution<bi::type::Integer_>(l_, u_)(rng);
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
  return std::binomial_distribution<bi::type::Integer_>(n_, ρ_)(rng);
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
  return std::negative_binomial_distribution<bi::type::Integer_>(k_, ρ_)(rng);
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
    return std::poisson_distribution<bi::type::Integer_>(λ_)(rng);
    }}
  } else {
    return 0;
  }
}

/**
 * Simulate a categorical variate.
 *
 * - ρ: Category probabilities.
 */
function simulate_categorical(ρ:Real[_]) -> Integer {
  assert length(ρ) > 0;

  u:Real <- simulate_uniform(0.0, 1.0);
  x:Integer <- 1;
  P:Real <- ρ[1];
  while (P < u) {
    assert x <= length(ρ);
    x <- x + 1;
    assert 0.0 <= ρ[x] && ρ[x] <= 1.0;
    P <- P + ρ[x];
    assert 0.0 <= P && P < 1.0 + 1.0e-6;
  }
  return x;
}

/**
 * Simulate a multinomial variate.
 *
 * - n: Number of trials.
 * - ρ: Category probabilities.
 *
 * This uses an O(N) implementation based on:
 *
 * Bentley, J. L. and J. B. Saxe (1979). Generating sorted lists of random
 * numbers. Technical Report 2450, Carnegie Mellon University, Computer
 * Science Department.
 */
function simulate_multinomial(n:Integer, ρ:Real[_]) -> Integer[_] {
  D:Integer <- length(ρ);
  R:Real[_] <- exclusive_prefix_sum(ρ);
  W:Real <- R[D] + ρ[D];

  lnMax:Real <- 0.0;
  j:Integer <- D;
  i:Integer <- n;
  u:Real;

  x:Integer[_] <- vector(0, D);
    
  while (i > 0) {
    u <- simulate_uniform(0.0, 1.0);
    lnMax <- lnMax + log(u)/i;
    u <- W*exp(lnMax);
    while (u < R[j]) {
      j <- j - 1;
    }
    x[j] <- x[j] + 1;
    i <- i - 1;
  }
  return x;
}

/**
 * Simulate a Dirichlet-categorical variate.
 *
 * - α: Concentrations.
 */
function simulate_dirichlet_categorical(α:Real[_]) -> Integer {
  return simulate_categorical(simulate_dirichlet(α));
}

/**
 * Simulate a Dirichlet-multinomial variate.
 *
 * - n: Number of trials.
 * - α: Concentrations.
 */
function simulate_dirichlet_multinomial(n:Integer, α:Real[_]) -> Integer[_] {
  return simulate_multinomial(n, simulate_dirichlet(α));
}

/**
 * Simulate a categorical variate with Chinese restaurant process prior.
 */
function simulate_crp_categorical(α:Real, θ:Real, n:Integer[_], N:Integer) -> Integer {
  assert N >= 0;
  assert sum(n) == N;

  k:Integer <- 0;
  K:Integer <- length(n);
  if (N == 0) {
    /* first component */
    k <- 1;
  } else {
    u:Real <- simulate_uniform(0.0, N + θ);
    U:Real <- K*α + θ;
    if (u < U) {
      /* new component */
      k <- K + 1;
    } else {
      /* existing component */
      while (k < K && u > U) {
        k <- k + 1;
        U <- U + n[k] - α;
      }
    }
  }
  return k;
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
  return std::uniform_real_distribution<bi::type::Real_>(l_, u_)(rng);
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
  return std::exponential_distribution<bi::type::Real_>(λ_)(rng);
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
    return std::normal_distribution<bi::type::Real_>(μ_, ::sqrt(σ2_))(rng);
    }}
  }
}

/**
 * Simulate a multivariate Gaussian variate.
 *
 * - μ: Mean.
 * - Σ: Covariance.
 */
function simulate_multivariate_gaussian(μ:Real[_], Σ:Real[_,_]) -> Real[_] {
  D:Integer <- length(μ);
  z:Real[D];
  for (d:Integer in 1..D) {
    z[d] <- simulate_gaussian(0.0, 1.0);
  }
  return μ + llt(Σ)*z;
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
    return std::lognormal_distribution<bi::type::Real_>(μ_, ::sqrt(σ2_))(rng);
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
  return std::gamma_distribution<bi::type::Real_>(k_, θ_)(rng);
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

/**
 * Simulate a Dirichlet variate.
 *
 * - α: Concentration.
 * - D: Number of dimensions.
 */
function simulate_dirichlet(α:Real, D:Integer) -> Real[_] {
  assert D >= 0;
  x:Real[D];
  z:Real <- 0.0;

  for (i:Integer in 1..D) {
    x[i] <- simulate_gamma(α, 1.0);
    z <- z + x[i];
  }
  z <- 1.0/z;
  for (i:Integer in 1..D) {
    x[i] <- z*x[i];
  }
  return x;
}
