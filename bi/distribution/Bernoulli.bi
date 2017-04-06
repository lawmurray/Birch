import math;
import random;
import assert;

/**
 * Bernoulli distribution.
 */
model Bernoulli {
  /**
   * Probability of a true result.
   */
  ρ:Real;
}

/**
 * Create.
 */
function Bernoulli(ρ:Real) -> m:Bernoulli {
  m.ρ <- ρ;
}

/**
 * Simulate.
 */
function (x:Boolean <~ m:Bernoulli) {
  cpp {{
  x = std::bernoulli_distribution(m.ρ)(rng);
  }}
}

/**
 * Observe.
 */
function (x:Boolean ~> m:Bernoulli) -> l:Real {
  if (x) {
    l <- log(m.ρ);
  } else {
    l <- log(1.0 - m.ρ);
  }
}
