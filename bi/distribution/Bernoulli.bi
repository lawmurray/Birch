import math;
import random;
import assert;

/**
 * Bernoulli distribution.
 */
class Bernoulli {
  /**
   * Probability of a true result.
   */
  ρ:Real;
}

/**
 * Create.
 */
function Bernoulli(ρ:Real) -> Bernoulli {
  m:Bernoulli;
  m.ρ <- ρ;
  return m;
}

/**
 * Simulate.
 */
function x:Boolean <~ m:Bernoulli {
  cpp {{
  x = std::bernoulli_distribution(m->ρ)(rng);
  }}
}

/**
 * Observe.
 */
function x:Boolean ~> m:Bernoulli -> Real {
  if (x) {
    return log(m.ρ);
  } else {
    return log(1.0 - m.ρ);
  }
}
