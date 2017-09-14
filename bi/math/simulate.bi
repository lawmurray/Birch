import math;

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
 * Simulate a Binomial variate.
 *
 * - n: Number of trials.
 * - ρ: Probability of a true result.
 */
function simulate_binomial(n:Integer, ρ:Real) -> Integer {
  assert n >= 0;
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp {{
  return std::binomial_distribution<bi::Integer_>(n_, ρ_)(rng);
  }}
}

/**
 * Simulate a Uniform variate.
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
  assert σ2 >= 0.0;
  if (σ2 == 0.0) {
    return μ;
  } else {
    cpp {{
    return std::normal_distribution<bi::Real_>(μ_, ::sqrt(σ2_))(rng);
    }}
  }
}

/**
 * Simulate a Gamma variate.
 *
 * - k: Shape.
 * - θ: Scale.
 */
function simulate_gamma(k:Real, θ:Real) -> Real {
  assert k > 0.0;
  assert θ > 0.0;
  cpp {{
  return std::gamma_distribution<bi::Real_>(k_, θ_)(rng);
  }}
}
