import math;
import random;
import assert;
import distribution.Uniform;

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
function ~m:Bernoulli -> x:Boolean {
  cpp {{
  x = static_cast<unsigned char>(std::bernoulli_distribution(m.ρ)(rng));
  }}
}

/**
 * Evaluate pmf.
 */
function x:Boolean ~> m:Bernoulli -> l:Real {
  if (x) {
    l <- m.ρ;
  } else {
    l <- 1.0 - m.ρ;
  }
}
