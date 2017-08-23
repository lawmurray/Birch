import math;

/**
 * Seed the pseudorandom number generator.
 *
 * `seed` Seed.
 */
function seed(s:Integer) {
  cpp {{
  rng.seed(s_);
  }}
}

/**
 * Simulate a Bernoulli variate.
 */
function random_bernoulli(ρ:Real) -> Boolean {
  assert 0.0 <= ρ && ρ <= 1.0;
  cpp {{
  return std::bernoulli_distribution(ρ_)(rng);
  }}
}

/**
 * Simulate a Uniform variate.
 */
function random_uniform(l:Real, u:Real) -> Real {
  assert l <= u;
  cpp {{
  return std::uniform_real_distribution<double>(l_, u_)(rng);
  }}
}

/**
 * Simulate a Gaussian variate.
 */
function random_gaussian(μ:Real, σ2:Real) -> Real {
  assert σ2 >= 0.0;
  if (σ2 == 0.0) {
    return μ;
  } else {
    cpp {{
    return std::normal_distribution<double>(μ_, ::sqrt(σ2_))(rng);
    }}
  }
}

/**
 * Simulate a Gamma variate.
 */
function random_gamma(k:Real, θ:Real) -> Real {
  assert k > 0.0;
  assert θ > 0.0;
  cpp {{
  return std::gamma_distribution<double>(k_, θ_)(rng);
  }}
}
