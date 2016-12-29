import math;
import random;
import assert;
import distribution.uniform;

/**
 * Bernoulli distribution.
 */
model Bernoulli(ρ1:Real) {
  /**
   * Probability of a true result.
   */
  ρ:Real <- ρ1;
}

/**
 * Simulate.
 */
function m:Bernoulli ~> x:Boolean {
  cpp {{
  x = static_cast<unsigned char>(std::bernoulli_distribution(m.ρ)(rng));
  }}
}

/**
 * Evaluate pmf.
 */
function x:Boolean ~ m:Bernoulli -> l:Real {
  if (x) {
    l <- m.ρ;
  } else {
    l <- 1.0 - m.ρ;
  }
}
